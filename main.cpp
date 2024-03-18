#include <iostream>
#include <filesystem>
#include <h5cpp/hdf5.hpp>

int main() {
    std::cout << "Hello, World!" << std::endl;
    std::filesystem::path path{"test.hdf"};
    hdf5::property::FileAccessList file_access_list;
    hdf5::property::FileCreationList file_creation_list;

    hdf5::file::File file = hdf5::file::create(path, hdf5::file::AccessFlags::ReadWrite, file_creation_list,
                                               file_access_list);
    return 0;
}
