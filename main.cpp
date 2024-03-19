#include <iostream>
#include <filesystem>
#include <h5cpp/hdf5.hpp>
#include <h5cpp/dataspace/simple.hpp>

int main() {
    std::cout << "Hello, World!" << std::endl;
    std::filesystem::path path{"./test.hdf"};

    auto file = hdf5::file::create(path, hdf5::file::AccessFlags::Exclusive, {}, {});
    auto root_group = file.root();

    // Write a scalar int value
    {
        auto const dataspace = hdf5::dataspace::Scalar();
        auto const data_type = hdf5::datatype::create<int32_t>();
        auto dataset = root_group.create_dataset("example_scalar1", data_type, dataspace);
        dataset.write(123, data_type, dataspace);
    }

    // Write a scalar string value
    {
        auto const dataspace = hdf5::dataspace::Scalar();
        auto const data_type = hdf5::datatype::create<std::string>();
        auto dataset = root_group.create_dataset("example_scalar2", data_type, dataspace);
        dataset.write(std::string{"hello"}, data_type, dataspace);
    }

    // Write a fixed size 1-D array of ints
    {
        std::vector<int32_t> data {1, 2, 3};
        auto const dataspace = hdf5::dataspace::Simple({3});
        auto const data_type = hdf5::datatype::create<int32_t>();
        auto dataset = root_group.create_dataset("example_simple1", data_type, dataspace);
        dataset.write(data, data_type, dataspace);
    }

    // Write a variable size 1-D array of ints
    // Requires a chunked dataset
    {
        std::vector<int32_t> data {1, 2, 3};
        auto const dataspace = hdf5::dataspace::Simple({0}, {hdf5::dataspace::Simple::unlimited});
        auto const data_type = hdf5::datatype::create<int32_t>();
        auto dataset = hdf5::node::ChunkedDataset(file.root(), hdf5::Path("example_simple2"), data_type, dataspace, {1024});

        dataset.extent(0, 3);
        dataset.write(data, hdf5::dataspace::Hyperslab{{0}, {3}});

        dataset.extent(0, 3);
        dataset.write(data, hdf5::dataspace::Hyperslab{{3}, {3}});
    }

    file.close();

    return 0;
}
