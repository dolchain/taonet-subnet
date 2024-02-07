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
            # Construct a CallMiners query. Send needed GPU size in GB
            synapse=CallMiners(needed_gpu=7),
            # All responses have the deserialize function called on them before returning.
            # You are encouraged to define your  own deserialization function.
            deserialize=True,
        )

        # Log the results for monitoring purposes.
        # bt.logging.info(f"Received responses: {volunteer_responses}")

        # Choose candidates among volunteers
        volunteer_uids, candidate_uids = get_candidate_uids(self, miner_uids=miner_uids, responses=volunteer_responses, peer_count=self.config.peer_count)
                
        if len(candidate_uids) == 0:
            continue

        candidate_responses = []
        # Make candidate miners to start training
        for i, uid in enumerate(candidate_uids):
            axon = self.metagraph.axons[uid]
            response = self.dendrite.query(
                axons=[self.metagraph.axons[uid]],  # Send the query to the selected axon
                synapse=StartMiners(peer_rank=i, peer_count=len(candidate_uids)),  # Use the corresponding synapse for this axon, Assign rank
                deserialize=True,  # Deserialize the response
            )
            candidate_responses.append(response)
        bt.logging.success(f"candidate_responses: {candidate_responses}")

        if all(candidate_responses):
            break

    bt.logging.success(f"peer_uids: {candidate_uids}")


    # # # TODO(developer): Define how the validator scores responses.
    # # # Adjust the scores based on responses from miners.
    # rewards = get_rewards(self, query=self.step, responses=responses)

    # bt.logging.info(f"Scored responses: {rewards}")
    # # # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    # self.update_scores(rewards, miner_uids)
