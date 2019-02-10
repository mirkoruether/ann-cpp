#include <utility>

#include "unambiguous_classification.h"

#include <chrono>
#include "mat_arr_math.h"

void annlib::tasks::unambiguous_classification::set_trainer(std::shared_ptr<annlib::sgd_trainer> t)
{
	trainer = std::move(t);
}

void annlib::tasks::unambiguous_classification::set_optimizer(std::shared_ptr<annlib::gradient_based_optimizer> o)
{
	optimizer = std::move(o);
}

void tasks::unambiguous_classification::set_custom_mb_builder(std::shared_ptr<mini_batch_builder> mbb)
{
	mb_builder = std::move(mbb);
}

void tasks::unambiguous_classification::set_data(tasks::classification_data data, unsigned mini_batch_size)
{
	mb_builder = std::make_shared<unambiguous_classification_mini_batch_builder>
		(std::move(data), mini_batch_size);
}

void tasks::unambiguous_classification::train(double epoch_count, const std::function<void(training_status)>* logger, unsigned int log_interval)
{
	trainer->train_epochs(optimizer.get(), *mb_builder, epoch_count, logger, log_interval);
}

std::vector<unsigned> get_max_indices(const mat_arr& ma)
{
	unsigned count = ma.count;
	unsigned row_col = ma.rows * ma.cols;

	std::vector<unsigned> result(count);
	for (unsigned mat_no = 0; mat_no < count; mat_no++)
	{
		fpt max_value = -1.0f * std::numeric_limits<fpt>::infinity();
		unsigned max_index = std::numeric_limits<unsigned>::max();

		const fpt* mat_start = ma.start() + mat_no * row_col;
		for (unsigned i = 0; i < row_col; i++)
		{
			fpt val = mat_start[i];
			if (std::isfinite(val) && val > max_value)
			{
				max_index = i;
				max_value = val;
			}
		}

		result[mat_no] = max_index;
	}

	return result;
}

mat_arr get_solution_net_output(std::vector<unsigned> labels, unsigned class_count)
{
	const auto count = static_cast<unsigned >( labels.size());
	mat_arr result(count, 1, class_count);
	fpt* start = result.start();
	for (unsigned i = 0; i < count; i++)
	{
		start[i * class_count + labels[i]] = 1.0f;
	}
	return result;
}

tasks::classification_test_result tasks::unambiguous_classification::test(tasks::classification_data data)
{
	mat_arr net_output = trainer->feed_forward(data.input);
	std::vector<unsigned> predicted_classes = get_max_indices(net_output);

	const unsigned count = data.input.count;
	const unsigned class_count = net_output.cols * net_output.rows;

	mat_arr total_confusion_matrix(1, class_count, class_count);
	fpt* c_mat_start = total_confusion_matrix.start();
	unsigned correct = 0;

	for (unsigned i = 0; i < count; i++)
	{
		unsigned predicted = predicted_classes[i];
		unsigned solution = data.labels[i];

		if (predicted >= class_count || solution >= class_count)
		{
			throw std::runtime_error("Unknown class");
		}

		if (predicted == solution)
		{
			correct++;
		}

		c_mat_start[solution * class_count + predicted] += 1.0f;
	}

	double total_costs = trainer->calculate_costs(net_output, get_solution_net_output(data.labels, class_count));

	return classification_test_result(count, correct, total_costs, total_confusion_matrix);
}

tasks::unambiguous_classification_mini_batch_builder::unambiguous_classification_mini_batch_builder(tasks::classification_data data,
                                                                                                    unsigned mini_batch_size)
	: mini_batch_builder(mini_batch_size, data.input.count), data(std::move(data))
{
	if (this->data.input.count != this->data.labels.size())
	{
		throw std::runtime_error("Input and labels differ in size");
	}
}

void tasks::unambiguous_classification_mini_batch_builder::build_mini_batch_internal(mat_arr* input_rv, mat_arr* solution_rv,
                                                                                     std::vector<unsigned> batch_indices) const
{
	mat_select_mats(data.input, batch_indices, input_rv);
	mat_set_all(0.0f, solution_rv);

	const unsigned row_col = solution_rv->rows * solution_rv->cols;
	fpt* s_start = solution_rv->start();
	for (unsigned i = 0; i < mini_batch_size; i++)
	{
		const unsigned label = data.labels[batch_indices[i]];
		s_start[i * row_col + label] = 1.0f;
	}
}

tasks::classification_test_result::classification_test_result(unsigned int count, unsigned int correct,
                                                              double total_costs, const mat_arr& total_confusion_matrix)
	: count(count), correct(correct), accuracy(static_cast<double>(correct) / count), total_costs(total_costs),
	  average_costs(total_costs / count), total_confusion_matrix(total_confusion_matrix),
	  average_confusion_matrix(mat_element_wise_div(total_confusion_matrix, static_cast<fpt>(count)))
{
}

tasks::classification_data::classification_data(mat_arr input, std::vector<unsigned int> labels)
	: input(std::move(input)), labels(std::move(labels))
{
	if (this->input.count != this->labels.size())
	{
		throw std::runtime_error("Input and labels differ in size");
	}
}
