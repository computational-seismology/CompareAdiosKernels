/******************************************************************************
 * Copyright 2017 Matthieu Lefebvre
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 *       limitations under the License.
 ******************************************************************************/

#ifndef COMPAREADIOSKERNELS_KERNEL_COMPARATOR_H
#define COMPAREADIOSKERNELS_KERNEL_COMPARATOR_H

#include "adios_reader.h"

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>

#include <iostream>
#include <functional>
#include <sstream>

#include <cmath>

namespace kernel_validation {
    /**
     * Perform the dot product of a vector with itself (x1*x1 + x2*x2 + x3*x3 + ...)
     * @tparam T Type of elements withn the vector. Typically ```float```
     * @param v The input vector
     * @return A value of type T containing the result
     */
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

    /**
     * Compute the difference between two vectors according to some metrics.
     *
     * Right now the metrics is ```ln ( u.u / v.v )```
     * It is designed to run with MPI and to perform a reduction within a communicator.
     * @tparam T The type of the vectors to compare
     * @param comm The MPI communicator in which operations are ran
     * @param u The first vector
     * @param v The second vector
     * @return The computed difference
     */
    template<typename T>
    T compute_diff(const mpi::communicator& comm, const std::vector<T>& u, const std::vector<T>& v)
    {
        T my_denominator = self_dot_product(u);
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

    /**
     * Class providing methods to compare matching arrays from two files
     */
    class KernelComparator {
    public:
        /**
         * Constructor
         * @param comm The MPI communicator in which operations are ran
         * @param ref_filename The name of the file containing the kernels acting as references
         * @param val_filename The name of the file containing the kernels to be assessed
         */
        KernelComparator(mpi::communicator comm,
                std::string ref_filename,
                std::string val_filename)
                :comm(comm),
                 ref_reader(ref_filename, comm),
                 val_reader(val_filename, comm) { }

        /**
         * Destructor
         */
        ~KernelComparator() { }

        /**
         * Compare a single value from two different ADIOS files
         * @param tolerance Accepted threshold when computing difference
         * @param var_name Name of the kernel to compute the difference for
         */
        void compare_single(float tolerance, std::string var_name);

        /**
         * Compare multiple values from two different ADIOS files
         * @param tolerance Accepted threshold when computing difference
         * @param kernel_list The list of kernel names to compute the difference for.
         */
        void compare_multiple(float tolerance, std::vector<std::string> kernel_list);

    private:
        mpi::communicator comm;
        ADIOSReader ref_reader;
        ADIOSReader val_reader;
    };

    ///////////////////////////////////////////////////////////////////////////
    /// Implementation

    void KernelComparator::compare_single(float tolerance, std::string var_name)
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

    void KernelComparator::compare_multiple(float tolerance, std::vector<std::string> kernel_list)
    {
        for (auto& var_name : kernel_list) {
            compare_single(tolerance, var_name);
        }
    }

}  // namespace kernel_validation
#endif //COMPAREADIOSKERNELS_KERNEL_COMPARATOR_H
