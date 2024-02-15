# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import time
import typing
import bittensor as bt

# Bittensor Miner Template:
import template

import taonet
from transformers import PreTrainedModel
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
from model.storage.model_metadata_store import ModelMetadataStore
from model.storage.remote_model_store import RemoteModelStore
from utilities.freegpu import get_free_gpu_memory


# import base miner class which takes care of most of the boilerplate
from template.base.miner import BaseMinerNeuron

class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        # TODO(developer): Anything specific to your use case you can do here

        # Indicates if miner status
        """
        waiting: Waiting for validator's call (could be ready when connection to vali is established)
        ready: Ready to work (could be working when turing is started)
        working: Working currently (could be waiting when turing is stopped)
        """
        self.status = 'waiting'

    async def forward(
        self, synapse: template.protocol.Dummy
    ) -> template.protocol.Dummy:
        """
        Processes the incoming 'Dummy' synapse by performing a predefined operation on the input data.
        This method should be replaced with actual logic relevant to the miner's purpose.

        Args:
            synapse (template.protocol.Dummy): The synapse object containing the 'dummy_input' data.

        Returns:
            template.protocol.Dummy: The synapse object with the 'dummy_output' field set to twice the 'dummy_input' value.

        The 'forward' function is a placeholder and should be overridden with logic that is appropriate for
        the miner's intended operation. This method demonstrates a basic transformation of input data.
        """
        # TODO(developer): Replace with actual implementation logic.
        synapse.dummy_output = synapse.dummy_input * 2
        return synapse

    async def blacklist(
        self, synapse: template.protocol.Dummy
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Dummy): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        # TODO(developer): Define how miners should blacklist requests.
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: template.protocol.Dummy) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Dummy): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority

    async def blacklist_call_miners(
        self, synapse: template.protocol.CallMiners
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)
    
    async def blacklist_init_miners(
        self, synapse: template.protocol.InitMiners
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def blacklist_start_miners(
        self, synapse: template.protocol.StartMiners
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)
    
    # Process income CallMiners Synapse
    async def call_miners(
        self, synapse: template.protocol.CallMiners
    ) -> template.protocol.CallMiners:
        # Get free gpu size
        free_memory_list = get_free_gpu_memory()
        # If free gpu is not available refuse the call
        if free_memory_list[0] < synapse.needed_gpu * 1024 or self.status != 'waiting':
            synapse.will_work = False
            return synapse
        # Will work if not working currently
        synapse.will_work = True
        return synapse

    # Process income InitMiners Synapse
    async def init_miners(
        self, synapse: template.protocol.InitMiners
    ) -> template.protocol.InitMiners:
        # Start work if not working currently
        vali_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        bt.logging.info(
            f'synapse from uid: {vali_uid}, {synapse.peer_rank}, {synapse.peer_count}')
        bt.logging.trace(
            f'loading model')

        # Load model.
        metadata_store = ChainModelMetadataStore(
            self.subtensor, self.wallet, self.config.netuid)
        remote_store = HuggingFaceModelStore()
        self.model: PreTrainedModel = await self.load_model_from_uid(
            vali_uid, self.config, self.metagraph, metadata_store, remote_store
        )
        bt.logging.trace(
            f'loaded model')

        self.rank = synapse.peer_rank
        self.peer_count = synapse.peer_count

        synapse.ready_to_work = True
        return synapse

    # Process income StartMiners Synapse
    async def start_miners(
        self, synapse: template.protocol.StartMiners
    ) -> template.protocol.StartMiners:
        # Start work if not working currently
        vali_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        bt.logging.info(
            f'synapse from uid: {vali_uid}, {synapse.master_addr}, {synapse.master_port}')
        
        self.master_addr = synapse.master_addr
        self.master_port = synapse.master_port
        self.status = 'ready'

        synapse.start_work = True
        return synapse


    async def load_model_from_uid(
        self,
        vali_uid: int,
        config: bt.config,
        metagraph: bt.metagraph,
        metadata_store: ModelMetadataStore,
        remote_model_store: RemoteModelStore,
    ) -> PreTrainedModel:

        # Initialize the model based on a passed uid.
        # Sync the state from the passed uid.
        model = await taonet.mining.load_remote_model(
            vali_uid,
            config.model_dir,
            metagraph,
            metadata_store,
            remote_model_store,
        )
        bt.logging.success(
            f"Training with model from uid: {config.load_uid}. Model={str(model)}"
        )
        return model


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            # bt.logging.info("Miner running...", time.time())
            time.sleep(30)
