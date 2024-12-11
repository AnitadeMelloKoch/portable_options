import os
import argparse

from evaluation.evaluators import AttentionEvaluatorClassifier

initiation_positive_files = [
    'resources/monte_images/climb_down_ladder_initiation_positive.npy'
]
initiation_negative_files = [
    'resources/monte_images/climb_down_ladder_initiation_negative.npy',
    'resources/monte_images/death_1.npy',
    'resources/monte_images/death_2.npy',
    'resources/monte_images/death_3.npy',
]

termination_positive_files = [
    'resources/monte_images/climb_down_ladder_termination_positive.npy',
]
termination_negative_files = [
    'resources/monte_images/climb_down_ladder_termination_negative.npy',
    'resources/monte_images/climb_down_ladder_initiation_positive.npy'
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--init_classifier_dir", type=str, required=True)
    parser.add_argument("--init_plot_dir", type=str, required=True)
    # parser.add_argument("--term_classifier_dir", type=str, required=True)
    # parser.add_argument("--term_plot_dir", type=str, required=True)

    args = parser.parse_args()

    evaluator_initiation = AttentionEvaluatorClassifier(
        args.init_plot_dir,
        args.init_classifier_dir
    )

    evaluator_initiation.add_true_from_files(initiation_positive_files)
    evaluator_initiation.add_false_from_files(initiation_negative_files)

    evaluator_initiation.evaluate_attentions(10)


    # evaluator_termination = AttentionEvaluatorClassifier(
    #     args.term_plot_dir,
    #     args.term_classifier_dir
    # )

    # evaluator_termination.add_true_from_files(termination_positive_files)
    # evaluator_termination.add_false_from_files(termination_negative_files)

    # evaluator_termination.evaluate_attentions(10)

    

