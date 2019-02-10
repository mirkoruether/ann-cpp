#ifndef ANN_CPP_CLASSIFICATION_H
#define ANN_CPP_CLASSIFICATION_H

#include <memory>
#include "sgd_trainer.h"

namespace annlib { namespace tasks
{
	struct classification_data
	{
		classification_data(mat_arr input, std::vector<unsigned int> labels);

		mat_arr input;
		std::vector<unsigned> labels;
	};

	struct classification_test_result
	{
		classification_test_result(unsigned int count, unsigned int correct, double total_costs, const mat_arr& total_confusion_matrix);

		unsigned count;

		unsigned correct;
		double accuracy;

		double total_costs;
		double average_costs;

		mat_arr total_confusion_matrix;
		mat_arr average_confusion_matrix;
	};

	class unambiguous_classification
	{
	private:
		std::shared_ptr<sgd_trainer> trainer;
		std::shared_ptr<gradient_based_optimizer> optimizer;
		std::shared_ptr<mini_batch_builder> mb_builder;

	public:
		unambiguous_classification() = default;

		void set_trainer(std::shared_ptr<sgd_trainer> t);

		template <typename Ty, typename... Tys>
		void set_trainer(Tys&& ... args)
		{
			set_trainer(std::make_shared<Ty>(std::forward<Tys>(args)...));
		}

		void set_optimizer(std::shared_ptr<gradient_based_optimizer> o);

		template <typename Ty, typename... Tys>
		void set_optimizer(Tys&& ... args)
		{
			set_optimizer(std::make_shared<Ty>(std::forward<Tys>(args)...));
		}

		void set_custom_mb_builder(std::shared_ptr<mini_batch_builder> mbb);

		template <typename Ty, typename... Tys>
		void set_custom_mb_builder(Tys&& ... args)
		{
			set_custom_mb_builder(std::make_shared<Ty>(std::forward<Tys>(args)...));
		}

		void set_data(classification_data data, unsigned mini_batch_size);

		void train(double epoch_count, const std::function<void(training_status)>* logger = nullptr, unsigned log_interval = 100);

		classification_test_result test(classification_data data);
	};

	class unambiguous_classification_mini_batch_builder : public mini_batch_builder
	{
	private:
		classification_data data;

	public:
		unambiguous_classification_mini_batch_builder(classification_data data, unsigned mini_batch_size);

	protected:
		void build_mini_batch_internal(mat_arr* input_rv, mat_arr* solution_rv, std::vector<unsigned> batch_indices) const override;
	};
}}

#endif //ANN_CPP_CLASSIFICATION_H
