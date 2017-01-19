#include <iostream>
#include <vector>
#include <numeric>
#include <functional>
#include <cmath>

#include "adios_read.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/operations.hpp>

// #include <boost/program_options/cmdline.hpp>

namespace mpi = boost::mpi;


// Log [int(reference-new)^2/int reference^2]

template <typename T>
T self_dot_product(std::vector<T>& v)
{
    // transform_reduce
    T res = 0;
    for (const auto x : v)
    {
        res += x*x;
    }
    return res;
}

template <typename T>
T diff_dot_product(std::vector<T>& u, std::vector<T>& v)
{
    T res = 0;
    for (auto it1 = u.begin(), it2 = v.begin(); it1 != u.end() && it2 != v.end(); ++it1, ++it2)
    {
        res += (*it1 - *it2) * (*it1 - *it2);
    }
    return res;
}

class ADIOSReader
{
    constexpr static int blocking_read = 1;
public:
    ADIOSReader(const char* fname, mpi::communicator& comm) {
        my_file = adios_read_open_file(fname, ADIOS_READ_METHOD_BP, comm);
    }
    ~ADIOSReader() {
        adios_read_close(my_file);
    }
    // TODO: pass only dir_name
    std::vector<float> schedule_read(const std::string& kernel_name, const std::string& region_name, int rank) {
        std::string dir_name = kernel_name + "_kl_" + region_name + "/";
        ADIOS_SELECTION* selection;
        selection = adios_selection_writeblock(rank);
        int local_dim_read;
        int offset_read;

        adios_schedule_read(my_file, selection, (dir_name + "local_dim").c_str(), 0, 1, &local_dim_read);
        adios_schedule_read(my_file, selection, (dir_name + "offset").c_str(), 0, 1, &offset_read);
        adios_perform_reads(my_file, blocking_read);

        uint64_t local_dim = local_dim_read; // up
        uint64_t offset = offset_read; // up
        std::vector<float> v(local_dim_read);

        selection = adios_selection_boundingbox(1, &offset, &local_dim);
        adios_schedule_read(my_file, selection, "rhonotprime_kl_crust_mantle/array", 0, 1, &v[0]);
        adios_perform_reads(my_file, blocking_read);

        //return std::move(v);
        return v;
    }

    ADIOS_FILE* my_file;
};


template <typename T>
float compute_diff(mpi::communicator comm, std::vector<T> u, std::vector<T> v)
{
    float my_denominator = self_dot_product(v);
    float full_denominator;
    mpi::reduce(comm, my_denominator, full_denominator, std::plus<float>(), 0);

    float my_numerator = diff_dot_product(v, v);
    float full_numerator;
    mpi::reduce(comm, my_numerator, full_numerator, std::plus<float>(), 0);

    float diff = -1;
    if (!comm.rank())
    {
        std::cerr << "full den: " << full_denominator << std::endl;
        std::cerr << "full num: " << full_numerator << std::endl;

        diff = std::sqrt(full_numerator / full_denominator);
    }
    return diff;
}


int main() {
    mpi::environment env;
    mpi::communicator world;

    if (!world.rank())
        std::cerr << "Running MPI with: " << world.size() << " processes." << std::endl;

    adios_read_init_method(ADIOS_READ_METHOD_BP, world, "");

    ADIOSReader reader("../kernels.bp", world);

    std::vector<std::pair<std::string, std::string>> kernel_list =
            {std::make_pair(std::string("rhonotprime"), std::string("crust_mantle"))
            ,std::make_pair(std::string("rho"), std::string("inner_core"))};

    for (auto& x : kernel_list) {
        std::cerr << "looking at kernel: " << x.first << std::endl;
        auto v = reader.schedule_read(x.first, x.second, world.rank());

        auto diff = compute_diff(world, v, v);
        if (!world.rank()) std::cerr << "Diff: " << diff << std::endl;
    }

    adios_read_finalize_method(ADIOS_READ_METHOD_BP);

    return 0;
}