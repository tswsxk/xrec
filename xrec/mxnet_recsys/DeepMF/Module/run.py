# coding: utf-8
# create by tongshiwei on 2019-9-1
import mxnet as mx
from longling import path_append
from longling.ML.MxnetHelper.toolkit.optimizer_cfg import get_update_steps


try:
    # for python module
    from .sym import get_net, get_bp_loss, fit_f, eval_f, net_viz
    from .etl import transform, etl, pseudo_data_iter, etl_eval
    from .configuration import Configuration, ConfigurationParser
except (ImportError, SystemError):  # pragma: no cover
    # for python script
    from sym import get_net, get_bp_loss, fit_f, eval_f, net_viz
    from etl import transform, etl, pseudo_data_iter, etl_eval
    from configuration import Configuration, ConfigurationParser


def get_lr_params(batches_per_epoch, lr, update_epoch=5, *args, **kwargs):
    return {
        "learning_rate": lr,
        "step": batches_per_epoch,
        "max_update_steps": get_update_steps(
            update_epoch=update_epoch,
            batches_per_epoch=batches_per_epoch,
        ),
    }


def numerical_check(_net, _cfg: Configuration, train_data, test_data, dump_result=False,
                    reporthook=None, final_reporthook=None):  # pragma: no cover
    ctx = _cfg.ctx
    batch_size = _cfg.batch_size

    _net.initialize(mx.init.Xavier(), ctx=ctx)

    bp_loss_f = get_bp_loss(**_cfg.loss_params)
    loss_function = {}
    loss_function.update(bp_loss_f)

    from longling.ML.MxnetHelper.glue import module
    from longling.ML.toolkit import EpochEvalFMT as Formatter
    from longling.ML.toolkit import MovingLoss
    from tqdm import tqdm

    loss_monitor = MovingLoss(loss_function)
    progress_monitor = tqdm
    if dump_result:
        from longling import config_logging
        validation_logger = config_logging(
            filename=path_append(_cfg.model_dir, "result.log"),
            logger="%s-validation" % _cfg.model_name,
            mode="w",
            log_format="%(message)s",
        )
        evaluation_formatter = Formatter(
            logger=validation_logger,
            dump_file=_cfg.validation_result_file,
            col=4,
        )
    else:
        evaluation_formatter = Formatter(col=4)

    # train check
    trainer = module.Module.get_trainer(
        _net, optimizer=_cfg.optimizer,
        optimizer_params=_cfg.optimizer_params,
        lr_params=None if _cfg.lr_lazy else _cfg.lr_params,
        select=_cfg.train_select
    )

    for epoch in range(_cfg.begin_epoch, _cfg.end_epoch):
        for i, batch_data in enumerate(progress_monitor(train_data, "Epoch: %s" % epoch)):
            fit_f(
                net=_net, batch_size=batch_size, batch_data=batch_data,
                trainer=trainer, bp_loss_f=bp_loss_f,
                loss_function=loss_function,
                loss_monitor=loss_monitor,
                ctx=ctx,
            )
        if _cfg.lr_lazy:
            print("reset trainer, batches per epoch: %s" % (i + 1))
            trainer = module.Module.get_trainer(
                _net, optimizer=_cfg.optimizer,
                optimizer_params=_cfg.optimizer_params,
                lr_params=get_lr_params(
                    batches_per_epoch=i + 1,
                    lr=_cfg.optimizer_params["learning_rate"],
                    update_epoch=_cfg.lr_params["update_epoch"]
                ),
                select=_cfg.train_select
            )
            _cfg.lr_lazy = False

        if epoch % 1 == 0:
            msg, data = evaluation_formatter(
                epoch=epoch,
                loss_name_value=dict(loss_monitor.items()),
                eval_name_value=eval_f(_net, test_data, ctx=ctx),
                extra_info=None,
                dump=dump_result,
                keep={"data", "msg"}
            )
            print(msg)
            if reporthook is not None:
                reporthook(data)
    if final_reporthook is not None:
        final_reporthook()


def pseudo_numerical_check(_net, _cfg):  # pragma: no cover
    train_data, eval_data = pseudo_data_iter(_cfg)
    numerical_check(_net, _cfg, train_data, eval_data, dump_result=False)


def train(train_fn, test_fn, reporthook=None, final_reporthook=None, **cfg_kwargs):  # pragma: no cover
    from longling.ML.toolkit.hyper_search import prepare_hyper_search

    cfg_kwargs, reporthook, final_reporthook, tag = prepare_hyper_search(
        cfg_kwargs, Configuration, reporthook, final_reporthook, primary_key="macro_auc"
    )

    _cfg = Configuration(**cfg_kwargs)
    _net = get_net(**_cfg.hyper_params)

    train_data = etl(_cfg.var2val(train_fn), params=_cfg)
    test_data = etl_eval(_cfg.var2val(test_fn), params=_cfg)

    numerical_check(_net, _cfg, train_data, test_data, dump_result=not tag, reporthook=reporthook,
                    final_reporthook=final_reporthook)


def sym_run(stage: (int, str) = "viz"):  # pragma: no cover
    if isinstance(stage, str):
        stage = {
            "viz": 0,
            "pseudo": 1,
            "real": 2,
            "cli": 3,
        }[stage]

    if stage <= 1:
        cfg = Configuration(
            hyper_params={
                "user_num": 1000,
                "item_num": 100,
                "vec_dim": 100,
                "op": "mlp"
            },
            eval_params={
                "unlabeled_value": 0,
                "k": [1, 3],
                "pointwise": True,
            }
        )
        net = get_net(**cfg.hyper_params)

        if stage == 0:
            # ############################## Net Visualization ###########################
            net_viz(net, cfg, False)
        else:
            # ############################## Pseudo Test #################################
            pseudo_numerical_check(net, cfg)

    elif stage == 2:
        # ################################# Simple Train ###############################
        import mxnet as mx
        train(
            "$data_dir/ml-1m/train.jsonl",
            "$data_dir/ml-1m/test.jsonl",
            hyper_params={
                "num_a": 6040,
                "num_b": 3900,
                "vec_dim": 128,
                "op": "linear"
            },
            root_data_dir="../../../../",
            optimizer_params={
                "learning_rate": 0.001
            },
            # ctx=[mx.gpu(3)],
            ctx=[mx.gpu(5)],
            batch_size=16,
            # optimizer_params={
            #     "learning_rate": 0.01
            # },
            # ctx=[mx.gpu(5), mx.gpu(6), mx.gpu(7), mx.gpu(8)],
            # batch_size=256,
        )

    elif stage == 3:
        # ################################# CLI ###########################
        cfg_parser = ConfigurationParser(Configuration, commands=[train])
        cfg_kwargs = cfg_parser()
        assert "subcommand" in cfg_kwargs
        subcommand = cfg_kwargs["subcommand"]
        del cfg_kwargs["subcommand"]
        print(cfg_kwargs)
        eval("%s" % subcommand)(**cfg_kwargs)

    else:
        raise TypeError


if __name__ == '__main__':  # pragma: no cover
    sym_run("real")
