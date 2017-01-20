#include "adios_reader.h"

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




template <typename T>
float compute_diff(mpi::communicator comm, std::vector<T> u, std::vector<T> v)
{
    float my_denominator = self_dot_product(v);
    float full_denominator;
    mpi::reduce(comm, my_denominator, full_denominator, std::plus<float>(), 0);

    float my_numerator = self_dot_product(v);
    float full_numerator;
    mpi::reduce(comm, my_numerator, full_numerator, std::plus<float>(), 0);

    float diff = -1;
    if (!comm.rank())
    {
        std::cerr << "full den: " << full_denominator << std::endl;
        std::cerr << "full num: " << full_numerator << std::endl;

        diff = std::log(full_numerator / full_denominator);
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

        std::string var_name = x.first + "_kl_" + x.second;

        if (!world.rank()) std::cerr << "looking at kernel: " << var_name << std::endl;
        auto v = reader.schedule_read<float>(var_name, world.rank());

        auto diff = compute_diff(world, v, v);
        if (!world.rank()) std::cerr << "Diff: " << diff << std::endl;
    }

    adios_read_finalize_method(ADIOS_READ_METHOD_BP);

    return 0;
}