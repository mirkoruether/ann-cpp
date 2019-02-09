#include "training_logger.h"

#include <sstream>
#include <mat_arr_math.h>

log::training_logger::training_logger(const std::string& file)
	: fs(file.c_str(), std::ios::out)
{
}

std::string build_buf_line(training_buffer* buf)
{
	std::ostringstream oss;
	for (unsigned i = 0; i < buf->layer_count(); i++)
	{
		layer_buffer* lbuf = buf->lbuf(i);

		//TODO
	}

	return oss.str();
}

void log::training_logger::log_status(training_status stat)
{
	const unsigned batch_no = stat.batch_no;
	const unsigned batch_count = stat.batch_count;

	if (batch_no == 0)
	{
		
	}

	if (batch_no < batch_count)
	{
		std::cout << "\r" << batch_no << "/" << batch_count
			<< " [" << unsigned(100.0 * (batch_no + 1) / batch_count) << "%]";

		fs << build_buf_line(stat.buf);
	}
	else
	{
		std::cout << "\r" << batch_count << "/" << batch_count << " [100%]" << std::endl;
	}
}
