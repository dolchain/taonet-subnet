import bittensor as bt

from template.protocol import CallMiners, StartMiners
from template.validator.reward import get_rewards
from template.utils.uids import get_all_uids, get_candidate_uids


async def start_train(self):
    """
    The start_miners function

    It is responsible for calling miners to establish a distributed training group and start work.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # TODO(developer): Define how the validator selects a miner to query, how often, etc.
    while True:

        # get_all_uids; to get all uids within the subnet
        miner_uids = get_all_uids(self, k=self.config.neuron.sample_size)

        # Get volunteers
        volunteer_responses = self.dendrite.query(
            # Send the query to selected miner axons in the network.
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            # Construct a CallMiners query.
            synapse=CallMiners(),
            # All responses have the deserialize function called on them before returning.
            # You are encouraged to define your  own deserialization function.
            deserialize=True,
        )

        # Log the results for monitoring purposes.
        # bt.logging.info(f"Received responses: {volunteer_responses}")

        # Choose candidates among volunteers
        volunteer_uids, candidate_uids = get_candidate_uids(self, miner_uids=miner_uids, responses=volunteer_responses, peer_count=self.config.peer_count)
                
        # Make candidate miners to start training
        candidate_responses = self.dendrite.query(
            # Send the query to selected miner axons in the network.
            axons=[self.metagraph.axons[uid] for uid in candidate_uids],
            # Construct a CallMiners query.
            synapse=StartMiners(),
            # All responses have the deserialize function called on them before returning.
            # You are encouraged to define your own deserialization function.
            deserialize=True,
        )
        if all(candidate_responses):
            break

    bt.logging.success(f"peer_uids: {candidate_uids}")


    # # # TODO(developer): Define how the validator scores responses.
    # # # Adjust the scores based on responses from miners.
    # rewards = get_rewards(self, query=self.step, responses=responses)

    # bt.logging.info(f"Scored responses: {rewards}")
    # # # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    # self.update_scores(rewards, miner_uids)
