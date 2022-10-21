import os
from collections import Counter
import pandas as pd

MaxMRRRank = 100


def load_reference_from_stream(f):
    """Load Reference reference relevant document
    Args:f (stream): stream to load.
    Returns:qids_to_relevant_documentids (dict): dictionary mapping from query_id (int) to relevant document (list of ints).
    """
    qids_to_relevant_documentids = {}
    for l in f:
        try:
            l = l.strip().split('\t')
            # qid = int(l[0])
            qid = l[0]
            if qid in qids_to_relevant_documentids:
                pass
            else:
                qids_to_relevant_documentids[qid] = []
            # qids_to_relevant_documentids[qid].append(l[2])
            qids_to_relevant_documentids[qid].append(l[1])
        except:
            raise IOError('\"%s\" is not valid format' % l)
    return qids_to_relevant_documentids


def load_reference(path_to_reference):
    """Load Reference reference relevant document
    Args:path_to_reference (str): path to a file to load.
    Returns:qids_to_relevant_documentids (dict): dictionary mapping from query_id (int) to relevant documents (list of ints).
    """
    with open(path_to_reference, 'r') as f:
        qids_to_relevant_documentids = load_reference_from_stream(f)
        # qids_to_relevant_documentids = load_candidate_from_stream(f)
    return qids_to_relevant_documentids


def validate_candidate_has_enough_ranking(qid_to_ranked_candidate_documents):
    for qid in qid_to_ranked_candidate_documents:
        if len(qid_to_ranked_candidate_documents[qid]) > MaxMRRRank:
            print('Too many documents ranked. Please Provide top 100 documents for qid:{}'.format(qid))


def load_candidate_from_stream(f):
    """Load candidate data from a stream.
    Args:f (stream): stream to load.
    Returns:qid_to_ranked_candidate_documents (dict): dictionary mapping from query_id (int) to a list of 1000 document ids(int) ranked by relevance and importance
    """
    qid_to_ranked_candidate_documents = {}
    for l in f:
        try:
            l = l.strip().split('\t')
            # qid = int(l[0])
            qid = l[0]
            did = l[1]
            rank = int(l[2])
            if qid in qid_to_ranked_candidate_documents:
                pass
            else:
                # By default, all PIDs in the list of 1000 are 0. Only override those that are given
                qid_to_ranked_candidate_documents[qid] = []
            qid_to_ranked_candidate_documents[qid].append((did, rank))
        except:
            raise IOError('\"%s\" is not valid format' % l)
    validate_candidate_has_enough_ranking(qid_to_ranked_candidate_documents)
    print('Quantity of Documents ranked for each query is as expected. Evaluating')
    return {qid: sorted(qid_to_ranked_candidate_documents[qid], key=lambda x: (x[1], x[0]), reverse=False) for qid in
            qid_to_ranked_candidate_documents}


def load_candidate(path_to_candidate):
    """Load candidate data from a file.
    Args:path_to_candidate (str): path to file to load.
    Returns:qid_to_ranked_candidate_documents (dict): dictionary mapping from query_id (int) to a list of 1000 document ids(int) ranked by relevance and importance
    """
    with open(path_to_candidate, 'r') as f:
        qid_to_ranked_candidate_documents = load_candidate_from_stream(f)
    return qid_to_ranked_candidate_documents


def quality_checks_qids(qids_to_relevant_documentids, qids_to_ranked_candidate_documents):
    """Perform quality checks on the dictionaries
    Args:
    p_qids_to_relevant_documentids (dict): dictionary of query-document mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_documents (dict): dictionary of query-document candidates
    Returns:
        bool,str: Boolean whether allowed, message to be shown in case of a problem
    """
    message = ''
    allowed = True

    # Create sets of the QIDs for the submitted and reference queries
    candidate_set = set(qids_to_ranked_candidate_documents.keys())
    ref_set = set(qids_to_relevant_documentids.keys())

    # Check that we do not have multiple documents per query
    for qid in qids_to_ranked_candidate_documents:
        # Remove all zeros from the candidates
        duplicate_pids = set(
            [item for item, count in Counter(qids_to_ranked_candidate_documents[qid]).items() if count > 1])

        if len(duplicate_pids - set([0])) > 0:
            message = "Cannot rank a document multiple times for a single query. QID={qid}, PID={pid}".format(
                qid=qid, pid=list(duplicate_pids)[0])
            allowed = False

    return allowed, message


def compute_metrics(qids_to_relevant_documentids, qids_to_ranked_candidate_documents, exclude_qids):
    """Compute MRR metric
    Args:
    p_qids_to_relevant_documentids (dict): dictionary of query-document mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_documents (dict): dictionary of query-document candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    all_scores = {}
    MRR = 0
    qids_with_relevant_documents = 0
    ranking = []
    for qid in qids_to_ranked_candidate_documents:
        if qid in qids_to_relevant_documentids and qid not in exclude_qids:
            ranking.append(0)
            target_pid = qids_to_relevant_documentids[qid]
            candidate_pid = qids_to_ranked_candidate_documents[qid]
            #if candidate_pid[0][0] == target_pid:
            #    hit +=1
            for i in range(0, len(candidate_pid)):
                if candidate_pid[i][0] in target_pid:
                    MRR += 1 / (i + 1)
                    ranking.pop()
                    ranking.append(i + 1)
                    break
    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")

    MRR = MRR / len(qids_to_relevant_documentids)
    all_scores['MRR @100'] = MRR
    all_scores['QueriesRanked'] = len(set(qids_to_ranked_candidate_documents) - exclude_qids)
    return all_scores


def compute_metrics_from_files(path_to_reference, path_to_candidate, exclude_qids, perform_checks=True):
    """Compute MRR metric
    Args:
    p_path_to_reference_file (str): path to reference file.
        Reference file should contain lines in the following format:
            QUERYID\tdocumentID
            Where documentID is a relevant document for a query. Note QUERYID can repeat on different lines with different documentIDs
    p_path_to_candidate_file (str): path to candidate file.
        Candidate file sould contain lines in the following format:
            QUERYID\tdocumentID1\tRank
            If a user wishes to use the TREC format please run the script with a -t flag at the end. If this flag is used the expected format is
            QUERYID\tITER\tDOCNO\tRANK\tSIM\tRUNID
            Where the values are separated by tabs and ranked in order of relevance
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """

    qids_to_relevant_documentids = load_reference(path_to_reference)
    qids_to_ranked_candidate_documents = load_candidate(path_to_candidate)
    if perform_checks:
        allowed, message = quality_checks_qids(qids_to_relevant_documentids, qids_to_ranked_candidate_documents)
        if message != '': print(message)

    return compute_metrics(qids_to_relevant_documentids, qids_to_ranked_candidate_documents, exclude_qids)


def load_exclude(path_to_exclude_folder):
    """Load QIDS for queries to exclude
    Args:
    path_to_exclude_folder (str): path to folder where exclude files are located
    Returns:
        set: a set with all qid's to exclude
    """
    qids = set()
    # List all files in a directory using os.listdir
    for a_file in os.listdir(path_to_exclude_folder):
        if os.path.isfile(os.path.join(path_to_exclude_folder, a_file)):
            with open(os.path.join(path_to_exclude_folder, a_file), 'r') as f:
                f.readline()  # header
                for l in f:
                    qids.add(int(l.split('\t')[0]))
    print("{} excluded qids loaded".format(len(qids)))
    return qids


def recall(args):

    q_gt, q_pred = {}, {}
    
    with open(args.res1_save_path, "r") as f:
        prev_q = ""
        for line in f.readlines():
            query, pred, gt, rank = line[:-1].split("\t")
            if args.trivia:
                if query != prev_q:
                    q_pred[query] = pred.split(",")
                    q_pred[query] = q_pred[query] 
                    prev_q = query
                if query in q_gt:
                    if len(q_gt[query]) <= 100:
                        q_gt[query].add(gt)
                else:
                    q_gt[query] = gt.split(",")
                    q_gt[query] = set(q_gt[query])
            else:
                if query != prev_q:
                    q_pred[query] = pred.split(",")
                    q_pred[query] = q_pred[query] 
                    prev_q = query
                if query in q_gt:
                    if len(q_gt[query]) <= 100:
                        q_gt[query].add(gt)
                else:
                    q_gt[query] = gt.split(",")
                    q_gt[query] = set(q_gt[query])

    print('#####################')
    if args.trivia:
        for i in args.recall_num:
            total = 0
            for q in q_pred:
                is_hit = 0
                for p in q_gt[q]:
                    if p in q_pred[q][:int(i)]:
                        is_hit = 1
                total += is_hit
            recall_avg = total / len(q_pred)
            print(f"recall@{i}: {recall_avg}")
            print('-------------------------')
    else:
        for i in args.recall_num:
            total = 0
            for q in q_pred:
                is_hit = 0
                for p in q_gt[q]:
                    if p in q_pred[q][:int(i)]:
                        is_hit = 1
                total += is_hit
            recall_avg = total / len(q_pred)
            print(f"recall@{i}: {recall_avg}")
            print('-------------------------')
    return recall_avg


def MRR100(args):
    mrr_total = 0
    query_num = 0
    with open(args.res1_save_path, "r") as f:
        for line in f.readlines():
            query, pred, gt, rank = line.split("\t")
            pred_list = pred.split(",")
            if gt in pred_list:
                rank = pred_list.index(gt) + 1
                mrr_total += 1 / rank
            query_num += 1

    mrr = mrr_total / query_num
    print('{}: {}'.format('MRR100', mrr))
    return mrr
