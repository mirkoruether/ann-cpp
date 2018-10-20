#ifndef NETINIT_H
#define NETINIT_H

#include "dmatrix.h"

using namespace linalg;

namespace annlib
{
	class NetInit
	{
	public:
		virtual ~NetInit() = default;
		virtual DRowVector initBiases(unsigned size) const = 0;
		virtual DMatrix initWeights(unsigned inputSize, unsigned outputSize) const = 0;
	};

	class GaussianInit : public NetInit
	{
	public:
		DRowVector initBiases(unsigned size) const override;
		DMatrix initWeights(unsigned inputSize, unsigned outputSize) const override;
	};

	class NormalizedGaussianInit : public GaussianInit
	{
	public:
		DMatrix initWeights(unsigned inputSize, unsigned outputSize) const override;
	};
}

#endif
