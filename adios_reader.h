//
// Created by Matthieu Lefebvre on 1/20/17.
//

#ifndef COMPAREADIOSKERNELS_ADIOS_READER_H
#define COMPAREADIOSKERNELS_ADIOS_READER_H

#include "adios_read.h"
#include <boost/mpi/communicator.hpp>
#include <vector>

namespace mpi = boost::mpi;

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
    template <typename T>
    std::vector<T> schedule_read(const std::string& var_name, int rank) {

        ADIOS_SELECTION* selection;
        selection = adios_selection_writeblock(rank);
        int local_dim_read;
        int offset_read;

        adios_schedule_read(my_file, selection, (var_name + "/local_dim").c_str(), 0, 1, &local_dim_read);
        adios_schedule_read(my_file, selection, (var_name + "/offset").c_str(), 0, 1, &offset_read);
        adios_perform_reads(my_file, blocking_read);

        uint64_t local_dim = local_dim_read; // up
        uint64_t offset = offset_read; // up
        std::vector<T> v(local_dim_read);

        selection = adios_selection_boundingbox(1, &offset, &local_dim);
        adios_schedule_read(my_file, selection, (var_name + "/array").c_str(), 0, 1, &v[0]);
        adios_perform_reads(my_file, blocking_read);

        //return std::move(v);
        return v;
    }

    ADIOS_FILE* my_file;
};


#endif //COMPAREADIOSKERNELS_ADIOS_READER_H
