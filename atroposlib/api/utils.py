from typing import Dict, List, Optional, Tuple


def grab_exact_from_heterogeneous_queue(
    queue: List[Dict[str, List]], batch_size: int
) -> Tuple[Optional[List], List]:
    """
    Grabs a batch of size batchsize from a queue of different sized items

    e.g. queue = [{"tokens": [[1, 2, 3],[4, 5, 6, 7, 8]]}, {"tokens": [[9, 10]]}]

    without going over the batchsize. This function will return a batch of size batchsize, and the new queue.

    Because all groups are a common denominator of the batchsize, and all groups are a power of 2,
    we can simplify a bit by assuming we can grab groups of groups to be equal to the maximum group size.
    Note that we cannot drop items from groups, so we must grab the entire group if we grab it.

    There may be a more efficient clearing mechanism by grouping these smaller groups heterogeneously, but
    forcing them all into powers of two groups is a simple way to ensure we can grab a batch of the correct size.

    :param queue:
    :param batch_size:
    :return: batch, new_queue
    """
    # check if we can even potentially grab a batch
    if sum(len(item["tokens"]) for item in queue) < batch_size:
        return None, queue
    # Get max batch size
    max_group_size = max(len(group["tokens"]) for group in queue)
    group_sizes = set(len(group["tokens"]) for group in queue)
    group_batching_storage = {i: [] for i in group_sizes}
    # pack the groups into [max_group_size // group_size] packs
    potential_batch = []
    for i, item in enumerate(queue):
        key = len(item["tokens"])
        group_batching_storage[key].append({"group": item, "indx": i})
        if len(group_batching_storage[key]) * key == max_group_size:
            potential_batch.extend(group_batching_storage[key])
            group_batching_storage[key] = []
    if (
        sum(len(grouped_items["group"]["tokens"]) for grouped_items in potential_batch)
        < batch_size
    ):
        return None, queue
    # we have a batch
    batch = []
    indxes_to_remove_from_queue = []
    for item in potential_batch:
        group = item["group"]
        indx = item["indx"]
        batch.append(group)
        indxes_to_remove_from_queue.append(indx)
        if sum(len(item["tokens"]) for item in batch) == batch_size:
            break
    if sum(len(item["tokens"]) for item in batch) != batch_size:
        return None, queue
    # remove the items from the queue
    new_queue = [
        item for i, item in enumerate(queue) if i not in indxes_to_remove_from_queue
    ]
    return batch, new_queue
