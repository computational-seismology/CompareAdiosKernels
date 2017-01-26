#ifndef COMPAREADIOSKERNELS_ADIOS_READER_H
#define COMPAREADIOSKERNELS_ADIOS_READER_H

#include "adios_read.h"
#include <boost/mpi/communicator.hpp>
#include <vector>
#include <exception>

namespace mpi = boost::mpi;


class ADIOSException : public std::exception {
  virtual const char* what() const throw()
  {
    return adios_errmsg(); 
  }
} adios_exception;


class ADIOSReader
{
    constexpr static int blocking_read = 1;
public:
    ADIOSReader(std::string filename, mpi::communicator& comm) {
        my_file = adios_read_open_file(filename.c_str(), ADIOS_READ_METHOD_BP, comm);
        if (adios_errno) throw adios_exception;
    }
    ~ADIOSReader() {
        adios_read_close(my_file);
        if (adios_errno) throw adios_exception;
    }
    template <typename T>
    std::vector<T> schedule_read(const std::string& var_name, int rank) {

        ADIOS_SELECTION* selection;
        selection = adios_selection_writeblock(rank);
        if (adios_errno) throw adios_exception;
        int local_dim_read;
        int offset_read;

        adios_schedule_read(my_file, selection, (var_name + "/local_dim").c_str(), 0, 1, &local_dim_read);
        if (adios_errno) throw adios_exception;
        adios_schedule_read(my_file, selection, (var_name + "/offset").c_str(), 0, 1, &offset_read);
        if (adios_errno) throw adios_exception;
        adios_perform_reads(my_file, blocking_read);
        if (adios_errno) throw adios_exception;

        uint64_t local_dim = local_dim_read; // up
        uint64_t offset = offset_read; // up
        std::vector<T> v(local_dim_read);

        selection = adios_selection_boundingbox(1, &offset, &local_dim);
        if (adios_errno) throw adios_exception;
        adios_schedule_read(my_file, selection, (var_name + "/array").c_str(), 0, 1, &v[0]);
        if (adios_errno) throw adios_exception;
        adios_perform_reads(my_file, blocking_read);
        if (adios_errno) throw adios_exception;

        //return std::move(v);
        return v;
    }

    ADIOS_FILE* my_file;
};


#endif //COMPAREADIOSKERNELS_ADIOS_READER_H
