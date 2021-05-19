from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from pipeline.utils.file.save import load as load_bin
import cv2
import os
from pprint import pprint
from loguru import logger


def visualize(predictor, on_im, using_metadata):

    im = cv2.imread(on_im)
    # format at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1],
                   metadata=using_metadata,
                   scale=1,
                   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   instance_mode=ColorMode.IMAGE_BW
                   )

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    visim = out.get_image()[:, :, ::-1]

    return visim, im


def run_visual_pred_on(images_dir, predictor, save_dir, vis_metadata, ext='jpg'):

    vis_metadata = load_bin(from_path=vis_metadata)  # train metadata

    write_dir = save_dir + f"eval_{images_dir[:-1].split('/')[-1]}" + '/'
    logger.debug(f"saving segmented images in {write_dir}")
    os.makedirs(write_dir, exist_ok=True)

    # check..
    logger.debug(repr(vis_metadata))

    for im_name in os.listdir(images_dir):
        if im_name.split('.')[-1].lower() == ext:
            # img = cv2.imread(images_dir+im_name)
            # output = predictor(img)
            # print(im_name,img.shape)
            # cv2.imwrite(
            #     f"{write_dir}onlymask_{im_name}",
            #     output
            # )
            visim, im = visualize(
                predictor=predictor,
                on_im=images_dir+im_name,
                using_metadata=vis_metadata)

            cv2.imwrite(
                f"{write_dir}segmented_{im_name}",
                visim
            )


def evaluate_kpis(model, cfg, kpi_out_dir, eval_dataset="dataset_holdout"):
    """ eval dataset must be registered"""

    #! todo: change out dir?
    evaluator = COCOEvaluator(
        eval_dataset, ("bbox", "segm"), False, output_dir=kpi_out_dir)
    eval_loader = build_detection_test_loader(cfg, eval_dataset)

    logger.debug("Calculating KPIs ...")
    pprint(inference_on_dataset(model, eval_loader, evaluator))
    # print(inference_on_dataset(trainer.model, eval_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`


if __name__ == "__main__":

    from pipeline.eval.det2.get_model import GetTrained
    model, cfg = GetTrained("ShopHoardingModel").fetch(thresh=0.5, cfg=True)

    evaluate(
        model=model,
        cfg=cfg)
