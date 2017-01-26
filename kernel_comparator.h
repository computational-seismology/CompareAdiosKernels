//
// Created by Matthieu Lefebvre on 1/20/17.
//

#ifndef COMPAREADIOSKERNELS_KERNEL_COMPARATOR_H
#define COMPAREADIOSKERNELS_KERNEL_COMPARATOR_H

#include "adios_reader.h"

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>

#include <iostream>
#include <functional>
#include <sstream>

#include <cmath>


template<typename T>
T self_dot_product(const std::vector<T>& v)
{
    // transform_reduce
    T res = 0;
    for (const auto x : v) {
        res += x*x;
    }
    return res;
}


template<typename T>
T compute_diff(const mpi::communicator& comm, const std::vector<T>& u, const std::vector<T>& v)
{
    T my_denominator = self_dot_product(v);
    T full_denominator;
    mpi::reduce(comm, my_denominator, full_denominator, std::plus<T>(), 0);

    T my_numerator = self_dot_product(v);
    T full_numerator;
    mpi::reduce(comm, my_numerator, full_numerator, std::plus<T>(), 0);

    T diff = -1;
    if (!comm.rank()) {
        diff = std::log(full_numerator/full_denominator);
    }
    return diff;
}


class KernelComparator {
public:
    KernelComparator(mpi::communicator comm, 
                     std::string ref_filename, 
                     std::string val_filename)
            : comm(comm),
              ref_reader(ref_filename, comm),
              val_reader(val_filename, comm) { }

    ~KernelComparator() { }

    void compare_single(float tolerance, std::string var_name)
    {
        if (!comm.rank()) std::cerr << "looking at kernel: " << var_name << std::endl;

        auto ref = ref_reader.schedule_read<float>(var_name, comm.rank());
        auto val = ref_reader.schedule_read<float>(var_name, comm.rank());
        auto diff = compute_diff(comm, ref, val);

        if (!comm.rank()) {
            if (std::isnan(diff)) {
                std::stringstream buffer;
                buffer << "[" << var_name << "] is NaN.";
                throw std::runtime_error(buffer.str());
            }
            if (diff>tolerance) {
                std::stringstream buffer;
                buffer << "[" << var_name << "] over tolerance: " << diff << " > " << tolerance << ".";
                throw std::runtime_error(buffer.str());
            }
        }
        if (!comm.rank()) {
            std::cerr << "Diff: " << diff << " ... moving on...\n";
        }
    }
    void compare_multiple(float tolerance, std::vector<std::string> kernel_list)
    {
        for (auto& var_name : kernel_list) {
            compare_single(tolerance, var_name);
        }
    }

private:
    mpi::communicator comm;
    ADIOSReader ref_reader;
    ADIOSReader val_reader;
};

#endif //COMPAREADIOSKERNELS_KERNEL_COMPARATOR_H
