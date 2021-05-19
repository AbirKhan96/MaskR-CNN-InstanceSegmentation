from pipeline.train.det2.trainer import Det2Trainer
from config import TrainConfig, DataConfig, ModelConfig, DataP


trainer = Det2Trainer(
    data=DataConfig.ShopHoardingDataset,
    model=ModelConfig.ShopHoardingModel,
    cfg=TrainConfig)

# process the json files into trainable data
trainer.prepare_data(DataPreparationConfig)

# generates out directory for assets' model after training
# and evaluating
trainer.start(
    resume=False,
    train_dataset=("dataset_train",),
    test_dataset=("dataset_test",))


if __name__ == "__main__":

    """
    Note: can run the below code independently
    after training is done!
    """

    from pipeline.eval.det2.get_model import GetTrained
    from pipeline.eval.det2.evaluation import evaluate_kpis, run_visual_pred_on
    from pipeline.utils.data_dict import DataDictCreator

    predictor, cfg = (
        GetTrained(ModelConfig.ShopHoardingModel.__name__)
        .fetch(thresh=0.01, cfg=True))

    # ideally `pred_on` should be 'holdout/'.
    # Yet, we can give anything there `train/` (or) `test/` or `holdout/`
    pred_on = 'holdout/'
    run_visual_pred_on(
        images_dir=DataConfig.ShopHoardingDataset.split_dataset_dir+pred_on,
        predictor=predictor,
        save_dir=DataConfig.ShopHoardingDataset.split_dataset_dir,
        vis_metadata=ModelConfig.ShopHoardingModel.save_dir+'metadata.bin'
    )

    #! todo: register `dataset_holdout` or others..
    #? doable with prpare_data(DataPreparationConfig)??  
    # can be `dataset_holdout` / `dataset_train` / `dataset_test`
    registered_dataset = "dataset_train"
    evaluate_kpis(
        eval_dataset=registered_dataset,
        model=predictor.model,
        cfg=cfg,
        kpi_out_dir=ModelConfig.ShopHoardingModel.save_dir + f'kpi_out/eval_{regestered_dataset}/')
