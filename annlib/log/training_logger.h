#ifndef TRAINING_LOGGER_H
#define TRAINING_LOGGER_H

#include "sgd_trainer.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace annlib;
using namespace linalg;

namespace annlib { namespace log
{
	class training_logger
	{
		std::ofstream fs;
	public:
		explicit training_logger(const std::string& file);

		void log_status(training_status stat);
	};
}}
#endif // TRAINING_LOGGER_H
