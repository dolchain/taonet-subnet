import math
import random
import torch
from datetime import timedelta
import bittensor as bt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import taonet
import threading

from taonet.dataset import SubsetFalconLoader
from traceback import print_exception
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore

def push_model(self, model, metadata_store, remote_store):
    bt.logging.debug(f'Pushing training model.')
    self.loop.run_until_complete(taonet.mining.push(
        model,
        self.config.hf_repo_id,
        self.wallet,
        metadata_store=metadata_store,
        remote_model_store=remote_store,
        uids = self.participate_uids if self.isTuring else [],
    ))
    bt.logging.debug(f'Pushed model')

def run(self, do_push: bool):
    try:
        bt.logging.trace(
            f'initing process group with params: {self.rank}, {self.peer_count}, {self.master_addr}, {self.master_port}')

        # Init process group
        dist.init_process_group(
            init_method=f"tcp://{self.master_addr}:{self.master_port}",
            backend='gloo',
            rank=self.rank,
            world_size=self.peer_count,
            timeout=timedelta(seconds=60)
        )
        bt.logging.trace(
            f'init processed')
    except Exception as err:
        bt.logging.debug('Error occured while initing process group', err)
        return

    try:        
        self.model = self.model.train()
        self.model = self.model.to(self.config.device)

        # build DistributedDataParallel Model
        ddp_model = DDP(self.model, device_ids=[0])

        # Build optimizer
        optimizer = torch.optim.AdamW(
            ddp_model.parameters(), lr=self.config.lr, weight_decay=0.01)

        # Start the training loop
        epoch_step = 0
        global_step = 0
        n_acc_steps = 0
        best_avg_loss = math.inf
        accumulation_steps = self.config.accumulation_steps

        last_upload_block = self.block
        metadata_store = ChainModelMetadataStore(
            self.subtensor, self.wallet, self.config.netuid)
        remote_store = HuggingFaceModelStore()      

        while True:
            # Initialize loss accumulator for the epoch
            epoch_loss = 0.0

            # Prepare the data loader with random pages for each epoch
            bt.logging.info(
                f"Loading {self.config.pages_per_epoch} pages for training this epoch"
            )
            max_pages = SubsetFalconLoader.max_pages
            random_pages = [
                random.randint(int(max_pages / self.peer_count * self.rank),
                                int(max_pages / self.peer_count * (self.rank + 1)))
                for _ in range(self.config.pages_per_epoch)
            ]
            loader = SubsetFalconLoader(
                batch_size=self.config.bs, sequence_length=self.config.sl, pages=random_pages
            )

            # Enumerate over the data loader
            n_batches = 0
            optimizer.zero_grad()  # Initialize gradients to zero

            for i, batch in enumerate(loader):
                # Move the input batch to the device
                inputs = batch.to(ddp_model.device)

                # Forward pass: compute the model output and loss
                outputs = ddp_model(inputs, labels=inputs)

                loss = outputs.loss / accumulation_steps  # Scale loss
                loss.backward()  # Accumulate gradients

                if (i + 1) % accumulation_steps == 0:
                    n_acc_steps += 1

                    # GRADIENT SHARE
                    optimizer.step()  # Perform a single optimization step
                    optimizer.zero_grad()  # Clear gradients
                    # Print log every time if it's a miner or one time per 5 steps if it's a vali
                    if not do_push or n_acc_steps % 5 == 0:
                        bt.logging.success(
                            f"Step: {n_acc_steps} loss: {outputs.loss.detach().item()}"
                        )

                torch.cuda.empty_cache()

                n_batches += 1
                global_step += 1
                epoch_loss += outputs.loss.detach().item()

            # Calculate the average loss for the epoch
            avg_loss = epoch_loss / n_batches

            # Log the average loss for the epoch
            bt.logging.success(
                f"Epoch: {epoch_step} average loss: {avg_loss}")
            epoch_step += 1

            # Check if the average loss of this epoch is the best we've seen so far
            if avg_loss < best_avg_loss:
                best_avg_loss = avg_loss  # Update the best average loss

                bt.logging.success(
                    f"New best average loss: {best_avg_loss}.")
                
                # If do_push save the model (means vali)
                if do_push:
                    # Upload model to Huggingface and publish chain if last uploaded block is over 100 blocks
                    if(self.block - last_upload_block > 100):
                        last_upload_block = self.block
                        push_thread = threading.Thread(target=push_model, args=(self, ddp_model.module, metadata_store, remote_store, ), daemon=False)
                        push_thread.start()

    except Exception as err:
        bt.logging.error("Error during training", str(err))
        bt.logging.debug(
            print_exception(type(err), err, err.__traceback__)
        )
        dist.destroy_process_group()
        self.isTuring = False
    finally:
        pass