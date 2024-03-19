#include <iostream>
#include <filesystem>
#include <h5cpp/hdf5.hpp>

int main() {
    std::cout << "Hello, World!" << std::endl;
    std::filesystem::path path{"./test.hdf"};

    auto file = hdf5::file::create(path, hdf5::file::AccessFlags::Exclusive, {}, {});
    auto root_group = file.root();

    // Write a scalar value
    {
        auto dataspace = hdf5::dataspace::Scalar();
        const hdf5::datatype::Integer data_type = hdf5::datatype::create<int32_t>();
        auto dataset = root_group.create_dataset("example_scalar1", data_type, dataspace);
        dataset.write(123, data_type, dataspace);
    }

    // Write an array of ints
    {
        std::vector<int32_t> data {1, 2, 3};
        auto dataspace = hdf5::dataspace::Simple({3});
        const hdf5::datatype::Integer data_type = hdf5::datatype::create<int32_t>();
        auto dataset = root_group.create_dataset("example_simple1", data_type, dataspace);
        dataset.write(data, data_type, dataspace);
    }

    file.close();

    return 0;
}
