#pragma once

#include "../utils/shape.hxx"
#include "./raii.cuh"
#include <string>

namespace raii {
    namespace adapter {
        template <DeviceArrValue T>
        // inherit to avoid wrapper boilerplate
        // just reexport some methods and constructors
        // protected inheritance to hide constructors
        class DeviceArr3D : protected DeviceArr<T, allocator::DeviceAllocatorD2> {
        protected:
            using cls      = DeviceArr3D<T>;
            using base_cls = DeviceArr<T, allocator::DeviceAllocatorD2>;

        public:
            using size_type  = cls::size_type;
            using value_type = cls::value_type;
            using shape_type = utils::Shape<3, size_type>;

        protected:
            const shape_type mShape;

            void
            checkShape(const cls& other) const {
                utils::checkShapesCompatibility(shape(), other.shape());
            }

        public:
            [[nodiscard]] DeviceArr3D(const cls& other)
            : base_cls(other), mShape{other.mShape} {
                checkShape(other.shape());
            }

            [[nodiscard]] DeviceArr3D(cls&& other)
            : base_cls(std::move(other)), mShape{other.mShape} {
                checkShape(other.shape());
            }

            [[nodiscard]] DeviceArr3D(const shape_type& shape)
            : base_cls(utils::shapeToSize(shape)), mShape{shape} {}

            [[nodiscard]] DeviceArr3D(const shape_type&& shape)
            : base_cls(utils::shapeToSize(shape)), mShape{std::move(shape)} {}

            [[nodiscard]] shape_type
            shape() const noexcept { return mShape; }

            [[nodiscard]] size_type
            sizeBytes() const noexcept { return base_cls::sizeBytes(); }

            [[nodiscard]] size_type
            shapeX() const noexcept { return std::get<0>(mShape); }

            [[nodiscard]] size_type
            shapeY() const noexcept { return std::get<1>(mShape); }

            [[nodiscard]] size_type
            shapeZ() const noexcept { return std::get<2>(mShape); }

            [[nodiscard]] device_ptr<T>
            data() const noexcept { return base_cls::data(); }

            void
            printFlat(
                const std::string& end = "\n",
                std::ostream&      out = std::cout) const {

                base_cls::print(end, out);
            }

            template <typename HostAlloc = std::pmr::polymorphic_allocator<T>>
            [[nodiscard]] inline std::vector<T, HostAlloc>
            toHost() const { return base_cls::toHost(); }

            // TODO reexport appropriate constructors and methods, static too
            // TODO adapt other methods
            // TODO add helper methods for 3D case
        };
    } // namespace adapter
} // namespace raii
