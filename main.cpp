#include <iostream>
#include <filesystem>
#include <h5cpp/hdf5.hpp>
#include <h5cpp/dataspace/simple.hpp>

    template<typename T>
    class Matrix {
    private:
        std::array<T, 9> data_{1,2,3,4,5,6,7,8,9};
    public:
        T *data() {
            return data_.data();
        }

        [[nodiscard]] const T *data() const {
            return data_.data();
        }
    };


namespace hdf5::dataspace {

    template<typename T>
    class TypeTrait<Matrix<T>> {
    public:
    using DataspaceType = Simple;

    static DataspaceType create(const Matrix<T> &) {
        std::cout << "create called\n";
        return Simple({3, 3});
    }

    const static DataspaceType &get(const Matrix<T> &) {
        const static DataspaceType &cref_ = Simple({3, 3});
        return cref_;
    }

    static void *ptr(Matrix<T> &value) {
        return reinterpret_cast<void *>(value.data());
    }

    static const void *cptr(const Matrix<T> &value) {
        return reinterpret_cast<const void *>(value.data());
    }
};

}

int main() {
    std::cout << "Hello, World!" << std::endl;
    std::filesystem::path path{"./test.hdf"};

    auto file = hdf5::file::create(path, hdf5::file::AccessFlags::Exclusive, {}, {});
    auto root_group = file.root();

    // Write a scalar int value
    {
        auto const dataspace = hdf5::dataspace::Scalar();
        auto const data_type = hdf5::datatype::create<int32_t>();
        auto dataset = root_group.create_dataset("scalar_int32", data_type, dataspace);
        dataset.write(123, data_type, dataspace);
    }

    // Write a scalar string value
    {
        auto const dataspace = hdf5::dataspace::Scalar();
        auto const data_type = hdf5::datatype::create<std::string>();
        auto dataset = root_group.create_dataset("scalar_string", data_type, dataspace);
        dataset.write(std::string{"hello"}, data_type, dataspace);
    }

    // Write a fixed size 1-D array of ints
    {
        std::vector<int32_t> data {1, 2, 3, 4, 5, 6};
        auto const dataspace = hdf5::dataspace::Simple({6});
        auto const data_type = hdf5::datatype::create<int32_t>();
        auto dataset = root_group.create_dataset("fixed_1d", data_type, dataspace);
        dataset.write(data, data_type, dataspace);
    }

    // Write a fixed size 1-D array of uint8s
    {
        std::vector<uint8_t> data {1, 2, 3, 4, 5, 6};
        auto const dataspace = hdf5::dataspace::Simple({6});
        auto const data_type = hdf5::datatype::create<uint8_t>();
        auto dataset = root_group.create_dataset("fixed_1d_uint8", data_type, dataspace);
        dataset.write(data, data_type, dataspace);
    }

    // Write a variable size 1-D array of ints
    // Requires a chunked dataset
    {
        std::vector<int32_t> data {1, 2, 3};
        auto const dataspace = hdf5::dataspace::Simple({0}, {hdf5::dataspace::Simple::unlimited});
        auto const data_type = hdf5::datatype::create<int32_t>();
        auto dataset = hdf5::node::ChunkedDataset(file.root(), hdf5::Path("stream_1d"), data_type, dataspace, {1024});

        dataset.extent(0, 3);
        dataset.write(data, hdf5::dataspace::Hyperslab{{0}, {3}});

        dataset.extent(0, 3);
        dataset.write(data, hdf5::dataspace::Hyperslab{{3}, {3}});
    }

    // Write a fixed size 2-D array of ints
    {
        // [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        std::vector<int32_t> data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        auto const dataspace = hdf5::dataspace::Simple({3, 4});
        auto const data_type = hdf5::datatype::create<int32_t>();
        auto dataset = root_group.create_dataset("fixed_2d", data_type, dataspace);
        dataset.write(data, data_type, dataspace);
    }

    // Write a variable size 2-D array of ints
    {
        std::vector<int32_t> data {1, 2, 3};
        auto const dataspace = hdf5::dataspace::Simple({0, 0}, {hdf5::dataspace::Simple::unlimited, hdf5::dataspace::Simple::unlimited});
        auto const data_type = hdf5::datatype::create<int32_t>();
        auto dataset = hdf5::node::ChunkedDataset(file.root(), hdf5::Path("stream_2d"), data_type, dataspace, {1024, 1024});

        dataset.extent(0, 1);
        dataset.extent(1, 3);
        dataset.write(data, hdf5::dataspace::Hyperslab{{0, 0}, {1, 3}});

        dataset.extent(0, 1);
        dataset.write(data, hdf5::dataspace::Hyperslab{{1, 0}, {1, 3}});

        // Increase the number of columns
        dataset.extent(0, 1);
        dataset.extent(1, 1);
        std::vector<int32_t> data2 {1, 2, 3, 4};
        dataset.write(data2, hdf5::dataspace::Hyperslab{{2, 0}, {1, 4}});
    }

    // Write a series of fixed size 2-D array of ints (e.g. an image stream)
    {
        // Writes two "images" of 3 rows x 4 columns
        hsize_t num_rows = 3;
        hsize_t num_cols = 4;
        std::vector<int32_t> data1 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::vector<int32_t> data2 {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
        auto const dataspace = hdf5::dataspace::Simple({0, num_rows, num_cols}, {hdf5::dataspace::Simple::unlimited, num_rows, num_cols});
        auto const data_type = hdf5::datatype::create<int32_t>();
        auto dataset = hdf5::node::ChunkedDataset(file.root(), hdf5::Path("image_stream"), data_type, dataspace, {1024, 3, 4});

        dataset.extent(0, 1);
        dataset.write(data1, hdf5::dataspace::Hyperslab{{0, 0, 0}, {1, num_rows, num_cols}});

        dataset.extent(0, 1);
        dataset.write(data2, hdf5::dataspace::Hyperslab{{1, 0, 0}, {1, num_rows, num_cols}});
    }

    // Custom type
    {
        // TODO: any value in this?
    }

    // Links
    {
        // Note: filewriter uses the underlying hdf library to do links - not sure why though...
        auto group = root_group.create_group("some_group");

        auto const dataspace = hdf5::dataspace::Scalar();
        auto const data_type = hdf5::datatype::create<int32_t>();
        auto dataset = group.create_dataset("scalar_int32", data_type, dataspace);
        dataset.write(456, data_type, dataspace);

        root_group.create_link(hdf5::Path("soft_link1"), group);
    }

    // Attributes
    {
        auto const dataspace = hdf5::dataspace::Scalar();
        auto const data_type = hdf5::datatype::create<int32_t>();
        auto dataset = root_group.create_dataset("scalar_dataset", data_type, dataspace);
        auto attribute = dataset.attributes.create<std::string>("author");
        attribute.write("some author");

        auto toplevel_attribute = root_group.attributes.create<std::string>("top-level attribute");
        toplevel_attribute.write("hello, world!");
    }

    file.close();

    return 0;
}
