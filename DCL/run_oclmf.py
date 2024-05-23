import argparse
from logging import getLogger
import torch

from recbole.config import Config
from recbole.trainer import Trainer
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed, set_color

from oclmf import OCLMF

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='amazon-books', help='The datasets can be: \
    ml-1m, yelp, amazon-books, gowalla-merged, alibaba')
    parser.add_argument('--config', type=str, default='', help='External config file name.')
    args, _ = parser.parse_known_args()

    #Config files
    args.config_file_list = [
        'properties/overall.yaml',
        'properties/OCLMF.yaml'
    ]
    if args.dataset in ['ml-1m', 'yelp', 'amazon-books', 'gowalla-merged', 'alibaba']:
        args.config_file_list.append(f'properties/{args.dataset}.yaml')
    # if args.config is not '':
    if args.config != '':
        args.config_file_list.append(args.config)

    config = Config(
        model=OCLMF,
        dataset=args.dataset, 
        config_file_list=args.config_file_list,
    )
    init_seed(config['seed'], config['reproducibility'])

    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = OCLMF(config, train_data.dataset).to(config['device'])
    logger.info(model)

    trainer = Trainer(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config['show_progress']
    )

    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
