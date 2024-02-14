import bittensor as bt

from template.protocol import StartMiners


async def start_train(self):
    """
    The start_train function

    It is responsible to broadcast to miners that the training is started.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # TODO(developer): Define how the validator selects a miner to query, how often, etc.
    # Broadcast to candidate uids to start training
    participate_responses = self.dendrite.query(
        # Send the query to selected miner axons in the network.
        axons=[self.metagraph.axons[uid] for uid in self.participate_uids],
        # Construct a CallMiners query. Send needed GPU size in GB
        synapse=StartMiners(master_addr=self.master_addr,
                            master_port=self.master_port),
        # All responses have the deserialize function called on them before returning.
        # You are encouraged to define your  own deserialization function.
        deserialize=True,
    )
    return all(participate_responses)

    # # # TODO(developer): Define how the validator scores responses.
    # # # Adjust the scores based on responses from miners.
    # rewards = get_rewards(self, query=self.step, responses=responses)

    # bt.logging.info(f"Scored responses: {rewards}")
    # # # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    # self.update_scores(rewards, miner_uids)
