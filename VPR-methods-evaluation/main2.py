import parser2
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

import vpr_models
from train_dataset import TrainDataset


def main(args):
    start_time = datetime.now()

 
    logger.remove()  # Remove possibly previously existing loggers
    log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(
        f"Testing with {args.method} with a {args.backbone} backbone and descriptors dimension {args.descriptors_dimension}"
    )
    logger.info(f"The outputs are being saved in {log_dir}")

    model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
    model = model.eval().to(args.device)

    train_ds = TrainDataset(args.database_folder, image_size=args.image_size)
    logger.info(f"Testing on {train_ds}")

    with torch.inference_mode():
        logger.debug("Extracting descriptors for Sparse Autoencoder")
        dataset = DataLoader(dataset=train_ds, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
        
        all_descriptors = np.empty((len(train_ds), args.descriptors_dimension), dtype="float32")
        for images, indices in tqdm(dataset):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy() 
            all_descriptors[indices.numpy(), :] = descriptors

        logger.debug(f"Saving final activations tensor in {log_dir}")
    if args.save_descriptors:
        logger.info(f"Saving the descriptors in {log_dir}")
        np.save(log_dir / "database_descriptors.npy", all_descriptors)
        

if __name__ == "__main__":
    args = parser.parse_arguments()
    main(args)