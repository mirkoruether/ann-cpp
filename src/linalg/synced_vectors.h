#ifndef SYNCED_VECTORS_H
#define SYNCED_VECTORS_H

#include <vector>
#include <iostream>

#ifdef LINALG_CUDA_SUPPORT
#include "dev_vector.h"
#include <cuda_runtime.h>
#endif

namespace linalg
{
	template <typename T>
	class synced_vectors
	{
	private:
		mutable bool host_synced = true;
		mutable bool dev_synced = true;

		mutable std::vector<T> host_vec;

#ifdef LINALG_CUDA_SUPPORT
		mutable dev_vector<T> dev_vec;
#endif

	public:
		explicit synced_vectors(size_t size)
			: host_vec(size)
#ifdef LINALG_CUDA_SUPPORT
			  , dev_vec(size)
#endif
		{
		}

		explicit synced_vectors(const std::vector<T>& data)
			: synced_vectors(data.size())
		{
			set(data);
		}

#ifdef LINALG_CUDA_SUPPORT
		explicit synced_vectors(const dev_vector<T>& data)
			: synced_vectors(data.size())
		{
			set(data);
		}
#endif

		explicit synced_vectors(const synced_vectors<T>& data, size_t offset, size_t size, bool try_device_copy = true)
			: synced_vectors(size)
		{
#ifdef LINALG_CUDA_SUPPORT
			if (try_device_copy)
			{
				my_cuda_memcp(dev_data(), data.dev_data() + offset, size, cudaMemcpyDeviceToDevice);
				return;
			}
#endif
			std::copy(data.host_data() + offset, data.host_data() + offset + size, host_data());
		}

		static bool host_available()
		{
			return true;
		}

		static bool dev_available()
		{
#ifdef LINALG_CUDA_SUPPORT
			return true;
#else
			return false;
#endif
		}

		void host_sync() const
		{
			if (!host_synced)
			{
				if (!dev_synced)
				{
					throw std::runtime_error("Illegal state");
				}
				device_to_host();
				host_synced = true;
			}
		}

		void dev_sync() const
		{
			if (!dev_synced)
			{
				if (!host_synced)
				{
					throw std::runtime_error("Illegal state");
				}
				host_to_device();
				dev_synced = true;
			}
		}

		void sync() const
		{
			host_sync();
			dev_sync();
		}

		T* host_data()
		{
#ifdef LINALG_CUDA_SUPPORT
			host_sync();
			dev_synced = false;
			return host_vec.data();
#else
			return host_vec.data();
#endif
		}

		const T* host_data() const
		{
#ifdef LINALG_CUDA_SUPPORT
			host_sync();
			return host_vec.data();
#else
			return host_vec.data();
#endif
		}

		T* dev_data()
		{
#ifdef LINALG_CUDA_SUPPORT
			dev_sync();
			host_synced = false;
			return dev_vec.data();
#else
			throw std::runtime_error("Not compiled with CUDA support");
#endif
		}

		const T* dev_data() const
		{
#ifdef LINALG_CUDA_SUPPORT
			dev_sync();
			return dev_vec.data();
#else
			throw std::runtime_error("Not compiled with CUDA support");
#endif
		}

		void set(const std::vector<T>& data)
		{
			if (data.size() != size())
			{
				throw std::runtime_error("Sizes do not fit");
			}

			dev_synced = false;
			host_synced = false;
			std::copy(data.data(), data.data() + size(), host_data());
			host_synced = true;
		}

#ifdef LINALG_CUDA_SUPPORT
		void set(const dev_vector<T>& data)
		{
			if (data.size() != size())
			{
				throw std::runtime_error("Sizes do not fit");
			}

			host_synced = false;
			dev_synced = false;
			my_cuda_memcp(dev_data(), data.data(), size(), cudaMemcpyDeviceToDevice);
			dev_synced = true;
		}
#endif

		size_t size() const
		{
			return host_vec.size();
		}

		void host_to_device(size_t offset, size_t length) const
		{
#ifdef LINALG_CUDA_SUPPORT
#ifdef LINALG_MONITOR_COPY
			std::cout << "host_to_device" << std::endl;
#endif
			my_cuda_memcp<T>(dev_vec.data() + offset, host_vec.data() + offset, length, cudaMemcpyHostToDevice);
#else
			//Do nothing
#endif
		}

		void host_to_device() const
		{
			host_to_device(0, size());
		}

		void device_to_host(size_t offset, size_t length) const
		{
#ifdef LINALG_CUDA_SUPPORT
#ifdef LINALG_MONITOR_COPY
			std::cout << "device_to_host" << std::endl;
#endif
			my_cuda_memcp<T>(host_vec.data() + offset, dev_vec.data() + offset, length, cudaMemcpyDeviceToHost);
#else
			//Do nothing
#endif
		}

		void device_to_host() const
		{
			device_to_host(0, size());
		}
	};
}

#endif
