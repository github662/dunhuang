import time

from options.train_options import TrainOptions
from dataloader.data_loader import dataloader
from model import create_model


if __name__ == '__main__':
    # 获取训练选项
    opt = TrainOptions().parse()
    # 创建数据集
    dataset = dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    print('Training images = %d' % dataset_size)
    # 创建模型
    model = create_model(opt)

    # 训练标志
    keep_training = True
    max_iteration = opt.niter + opt.niter_decay
    epoch = 0
    total_iteration = opt.iter_count

    # 训练过程
    while keep_training:
        epoch_start_time = time.time()
        epoch += 1
        print('\nTraining epoch: %d' % epoch)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_iteration += 1
            model.set_input(data)
            model.optimize_parameters()

            import os

            if total_iteration % opt.print_freq == 0:
                losses = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize

                # 格式化日志内容
                log_line = 'Epoch: %d, Iteration: %d, Losses: %s, Time: %.3f' % (
                    epoch, total_iteration, str(losses), t
                )

                # 打印到终端
                print(log_line)

                # 构造日志文件路径：checkpoints_dir/name/loss_log.txt
                log_dir = os.path.join(opt.checkpoints_dir, opt.name)
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, "loss_log.txt")

                # 写入日志文件
                with open(log_path, "a") as f:
                    f.write(log_line + "\n")

            # 每<save_latest_freq>次保存模型
            if total_iteration % opt.save_latest_freq == 0:
                print('Saving the latest model (epoch %d, total_steps %d)' % (epoch, total_iteration))
                model.save_networks('latest')

            # 每<save_iters_freq>次保存模型
            if total_iteration % opt.save_iters_freq == 0:
                print('Saving the model of iterations %d' % total_iteration)
                model.save_networks(total_iteration)

            # 达到最大迭代次数时停止训练
            if total_iteration > max_iteration:
                keep_training = False
                break

        # 更新学习率
        model.update_learning_rate()

        print('\nEnd training')
