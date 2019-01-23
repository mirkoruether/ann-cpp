#ifndef DEV_VECTOR_H
#define DEV_VECTOR_H

#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>

namespace linalg
{
	template <typename T>
	void my_cuda_memcp(T* dst, const T* src, size_t length, cudaMemcpyKind kind)
	{
		if(cudaDeviceSynchronize() != cudaSuccess)
		{
			throw std::runtime_error("error on previous operation");
		}

		if(dst == nullptr || src == nullptr)
		{
			throw std::runtime_error("nullptr");
		}

		const cudaError_t result = cudaMemcpy(dst, src,
		                                      length * sizeof(T),
		                                      kind);
		if (result != cudaSuccess)
		{
			throw std::runtime_error("Failed to copy");
		}
	}

	template <typename T>
	class dev_vector
	{
	private:
		T* start;
		size_t count;

		void allocate(size_t pcount)
		{
			const cudaError_t result = cudaMalloc(&start, pcount * sizeof(T));
			if (result != cudaSuccess)
			{
				start = nullptr;
				count = 0;
				throw std::runtime_error("Failed to allocate device memory");
			}
			count = pcount;
		}

		void free()
		{
			if (start != nullptr)
			{
				cudaFree(start);
				count = 0;
				start = nullptr;
			}
		}

	public:
		explicit dev_vector(const std::vector<T>& vec)
			: dev_vector(vec.size())
		{
			my_cuda_memcp(start, vec.data(), count, cudaMemcpyHostToDevice);
		}

		explicit dev_vector(const dev_vector<T>& vec)
			: dev_vector(vec.size())
		{
			my_cuda_memcp(start, vec.data(), count, cudaMemcpyDeviceToDevice);
		}

		explicit dev_vector(size_t size)
		{
			allocate(size);
		}

		~dev_vector()
		{
			free();
		}

		const T* data() const
		{
			return start;
		}

		T* data()
		{
			return start;
		}

		size_t size() const
		{
			return count;
		}

		std::vector<T> to_vector()
		{
			std::vector<T> vec(count);
			my_cuda_memcp(vec.data(), start, count, cudaMemcpyDeviceToHost);
			return vec;
		}
	};
}
#endif
