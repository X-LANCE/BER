import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from scipy import optimize
from scipy.io import wavfile as wav
import numpy as np
from argparse import ArgumentParser
from tabulate import tabulate
import concurrent.futures as cf


def get_rttm_dict(rttm_file):
    """build meta information from single rttm file path.
    Args:
        rttm_file: rttm file path
    Returns:
        rttm_dict: {'filename': {'spkname':[(start,end),...]}}
    """
    rttm_dict = dict()
    for line in open(rttm_file).readlines():
        items = line.replace("\n", "").split()
        filename, start_time, duration, spk_name = items[1], float(items[3]), float(items[4]), items[7]
        end_time = start_time + duration

        if filename not in rttm_dict.keys():
            rttm_dict[filename] = dict()
        if spk_name not in rttm_dict[filename].keys():
            rttm_dict[filename][spk_name] = []
        rttm_dict[filename][spk_name].append((start_time,end_time))
    return rttm_dict


def merge_duplicated_segment(segments):
    """merge overlapped segments
    Args:
        segments: [(start, end), ...]
    Returns:
        segments: [(start, end), ...]
    """

    if len(segments) <= 1:
        return segments

    segments.sort()

    segments_no_ovl = [segments[0]]
    for i in range(1,len(segments)):
        last_start, last_end, current_start, current_end = segments_no_ovl[-1][0], segments_no_ovl[-1][1], segments[i][0], segments[i][1]
        overlap = last_end - current_start
        if overlap >= 0 :
            merged_seg = (last_start, current_end)
            segments_no_ovl.pop()
            segments_no_ovl.append(merged_seg)
        else:
            segments_no_ovl.append(segments[i]) # no overlap
    return segments_no_ovl


def read_rttm_files(ref_rttm_file, hyp_rttm_file):
    """build meta information from ref and hyp rttm file path.
    Args:
        ref_rttm_file, hyp_rttm_file: rttm file path
    Returns:
        rttm_dict: {'filename': { 'hyp' or 'ref': [(spkname, start, end)]}}
    """
    ref_rttm_dict, hyp_rttm_dict = get_rttm_dict(ref_rttm_file), get_rttm_dict(hyp_rttm_file)
    rttm_dict = dict()

    for filename in ref_rttm_dict.keys():
        if filename not in rttm_dict.keys():
            rttm_dict[filename] = dict()
            if 'ref' not in rttm_dict[filename].keys():
                rttm_dict[filename]['ref'] = []
        for spk_name in ref_rttm_dict[filename]:
            ref_rttm_dict[filename][spk_name] = merge_duplicated_segment(ref_rttm_dict[filename][spk_name])

            for start, end in ref_rttm_dict[filename][spk_name]:
                rttm_dict[filename]['ref'].append((spk_name, start, end))

    for filename in hyp_rttm_dict.keys():
        assert filename in rttm_dict.keys()
        if 'hyp' not in rttm_dict[filename].keys():
            rttm_dict[filename]['hyp'] = []
        for spk_name in hyp_rttm_dict[filename]:
            hyp_rttm_dict[filename][spk_name] = merge_duplicated_segment(hyp_rttm_dict[filename][spk_name])
            for start, end in hyp_rttm_dict[filename][spk_name]:
                rttm_dict[filename]['hyp'].append((spk_name, start, end))

    return rttm_dict


def compute_intersection_length(A, B):
    """Compute the intersection length of two tuples.
    Args:
        A: a (speaker, start, end) tuple of type (string, float, float)
        B: a (speaker, start, end) tuple of type (string, float, float)
    Returns:
        a float number of the intersection between `A` and `B`
    """
    max_start = max(A[1], B[1])
    min_end = min(A[2], B[2])
    return max(0.0, min_end - max_start)


def build_speaker_index(hyp):
    """Build the index for the speaker names.
    Args:
        hyp: a list of tuples, where each tuple is (speaker, start, end)
            of type (string, float, float)
    Returns:
        a dict from speaker to integer
    """
    speaker_set = sorted({element[0] for element in hyp})
    index = {speaker: i for i, speaker in enumerate(speaker_set)}
    return index


def build_speaker_list(ref, hyp):
    """Build the index for the speakers.
    Args:
        ref: a list of tuples for the ground truth, where each tuple is
            (speaker, start, end) of type (string, float, float)
        hyp: a list of tuples for the diarization result hypothesis, same type
            as `ref`
    Returns:
        ref_index, hyp_index: mappings from speaker name to index for ref and hyp
        index_ref, index_hyp: the revsered mappings
    """
    ref_index = build_speaker_index(ref)
    hyp_index = build_speaker_index(hyp)

    index_ref = dict()
    for key in ref_index:
        index_ref[ref_index[key]] = key

    index_hyp = dict()
    for key in hyp_index:
        index_hyp[hyp_index[key]] = key

    return ref_index, hyp_index, index_ref, index_hyp


def build_cost_matrix(ref, hyp, ref_index, hyp_index):
    """Build the cost matrix.
    Args:
        ref: a list of tuples for the ground truth, where each tuple is
            (speaker, start, end) of type (string, float, float)
        hyp: a list of tuples for the diarization result hypothesis, same type
            as `ref`
    Returns:
        a 2-dim numpy array, whose element (i, j) is the overlap between
            `i`th reference speaker and `j`th hypothesis speaker
    """

    cost_matrix = np.zeros((len(ref_index), len(hyp_index)))
    for ref_element in ref:
        for hyp_element in hyp:
            i = ref_index[ref_element[0]]
            j = hyp_index[hyp_element[0]]
            cost_matrix[i, j] += compute_intersection_length(
                ref_element, hyp_element)
    return cost_matrix


def build_connection_matrix(ref_segments, hyp_segments):
    """Build the connection matrix.
    Args:
        ref: a list of tuples for the ground truth, where each tuple is
            (speaker, start, end) of type (string, float, float)
        hyp: a list of tuples for the diarization result hypothesis, same type
            as `ref`
    Returns:
        a 2-dim numpy array, whose element (i, j) = 1 means there exists a overlap between
            `i`th reference speaker and `j`th hypothesis speaker
    """
    connection_matrix = np.zeros((len(ref_segments), len(hyp_segments)))

    for i, ref_segment in enumerate(ref_segments):
        for j, hyp_segment in enumerate(hyp_segments):
            max_start = max(ref_segment[0], hyp_segment[0])
            min_end = min(ref_segment[1], hyp_segment[1])
            intersection = max(0.0, min_end - max_start) # two segments are overlapped
            if intersection > 0:
                connection_matrix[i][j] = 1
    return connection_matrix


def test_connectivity_by_dfs(connection_matrix, current_pair, i ,j):
    """Get all nodes in a connected sub-graph via depth-first traversal
    Args:
        connection_matrix: a 2-dim numpy array, whose element (i, j) = 1 means there exists a overlap between
            `i`th reference speaker and `j`th hypothesis speaker
        hyp: a list of tuples for the diarization result hypothesis, same type
            as `ref`
    Returns:
        N/A
    """
    if i < 0 or j < 0 or i >= len(connection_matrix) or j >= len(connection_matrix[0]):
        return
    if connection_matrix[i][j] != 1:
        return
    current_pair.append((i,j))
    connection_matrix[i][j] = 0 # marked as visited
    test_connectivity_by_dfs(connection_matrix, current_pair, i-1 ,j) # left
    test_connectivity_by_dfs(connection_matrix, current_pair, i+1 ,j) # right
    test_connectivity_by_dfs(connection_matrix, current_pair, i ,j-1) # up
    test_connectivity_by_dfs(connection_matrix, current_pair, i ,j+1) # down


def get_all_connected_graphs(connection_matrix):
    """Get all connected sub-graph in connection_matrix
    Args:
        connection_matrix: a 2-dim numpy array, whose element (i, j) = 1 means there exists a overlap between
            `i`th reference speaker and `j`th hypothesis speaker
        hyp: a list of tuples for the diarization result hypothesis, same type
            as `ref`
    Returns:
        total_pair: matched node list, [[(i,j), ... ], ... ] total_pair length means the number of sub-graphs. Its item is a node matched list.
    """
    total_pair = []

    for i in range(len(connection_matrix)):
        for j in range(len(connection_matrix[0])):
            if connection_matrix[i][j] == 1:
                current_pair = []
                test_connectivity_by_dfs(connection_matrix, current_pair, i ,j)
                total_pair.append(current_pair)

    return total_pair


def iou_of_pair_segments(segments_a, segments_b):
    """Calculate the IoU of between segments_a listsand segments_b list
    Args:
        segments_a: a list with item (start, end) of type (float, float)
        segments_b: a list with item (start, end) of type (float, float)
    Returns:
        IoU: IoU value, also called jaccard index
        segments_a_total: total length of segment a
        segments_b_total: total length of segment b
        intersection: the numerator of IoU
        union: the denominator of IoU
    """
    segments_a_total, segments_b_total, intersection = 0, 0, 0

    for start_a, end_a in segments_a:
        for start_b, end_b in segments_b:
            intersection += max(0.0, min(end_a, end_b) - max(start_a, start_b))

    for start_a, end_a in segments_a:
        segments_a_total += end_a - start_a

    for start_b, end_b in segments_b:
        segments_b_total += end_b - start_b

    union = segments_a_total + segments_b_total - intersection

    IoU = intersection/union

    return IoU, segments_a_total, segments_b_total, intersection, union


def get_dynamic_iou(single_ref_segs, single_hyp_segs, lb=0.5, collar=0.5):
    """Get dynamic iou for segments with varying length because the fixed IoU (like 0.5) is too loose for long utterance
    Args:
        single_ref_segs: a list with item (start, end) of type (float, float)
        single_hyp_segs: a list with item (start, end) of type (float, float)
        lb: lower bound of IoU, the default is 0.5
        collar: collar is used here to adjust the border flexibility for segment-level errors. We follow DER and the default value is 0.25*2 (Two means: before and after the boundary)
    Returns:
        IoU: IoU value, also called jaccard index
        segments_a_total: total length of segment a
        segments_b_total: total length of segment b
        intersection: the numerator of IoU
        union: the denominator of IoU
    """
    total_duration = sum([end-start for start, end in single_ref_segs])
    return max((total_duration-2*collar*len(single_ref_segs))/(total_duration+2*collar*len(single_ref_segs)),lb) # multiplying by two means segment start and end


def get_fa_ms(ref_segments, hyp_segments, precison = 100):
    """Calculate false alarm and missed speech duration
    Args:
        ref_segments: a sorted list with item (start, end) of type (float, float)
        hyp_segments: a sorted list with item (start, end) of type (float, float)
        precison: adjust the precision for duration and the default is 100 which means the time resolution is 0.01 seconds
    Returns:
        fa_duration: false alarm duration in seconds, used as one of the numerator of duration error rate
        ms_duration: missed speech duration in seconds, used as one of the numerator of duration error rate
        intersection: intersection duration in seconds, used in JER but not used in our metric
        ref_duration: reference duration in seconds, used as the denominator of duration error rate
        union_duration:  union duration in seconds, used in JER but not used in our metric
    """


    max_len = max(ref_segments[-1][-1], hyp_segments[-1][-1])

    ref_vector = np.zeros(int(round(max_len*precison)+1)).astype("bool")
    hyp_vector = np.zeros(int(round(max_len*precison)+1)).astype("bool")

    for start, end in ref_segments:
        ref_vector[int(round(start*precison)):int(round(end*precison))] = True

    for start, end in hyp_segments:
        hyp_vector[int(round(start*precison)):int(round(end*precison))] = True

    fa_duration_vector = ~ ref_vector & hyp_vector
    ms_duration_vector = ref_vector & ~ hyp_vector
    intersection_vector = ref_vector & hyp_vector
    union_vector = ref_vector | hyp_vector

    assert sum([sum(ms_duration_vector), sum(intersection_vector)]) == sum(ref_vector)

    fa_duration = sum(fa_duration_vector)/precison
    ms_duration = sum(ms_duration_vector)/precison
    intersection = sum(intersection_vector)/precison
    ref_duration= sum(ref_vector)/precison
    union_duration = sum(union_vector)/precison
    return (fa_duration, ms_duration, intersection, ref_duration, union_duration)


def harmonic_mean(value1,value2, eps = 1e-6 ):
    return 2 / (1/(value1+eps) + 1/(value2+eps)) - eps


def get_statistics_for_each_file(rttm_dict_item):

    filename, file_rttm_dict = rttm_dict_item

    # the detailed print information for each file is stored in this list
    detailed_result_list = []

    file_bder_dict = dict()
    file_bder_dict['optimal_mapping'] = dict()
    file_bder_dict['optimal_mapping']['r2h'] = dict() # ref to hyp speaker mapping after optimal_mapping
    file_bder_dict['optimal_mapping']['h2r'] = dict() # hyp to ref
    file_bder_dict['optimal_mapping']['r_ms_id'] = set() # not matched ref id, also called missed speaker
    file_bder_dict['optimal_mapping']['h_ms_id'] = set() # not matched hyp id, also called false alarm speaker
    file_bder_dict['optimal_mapping']['r_ms_case'] = []  # not matched ref case
    file_bder_dict['optimal_mapping']['h_ms_case'] = []  # not matched hyp case

    file_bder_dict['ser'] = dict()
    file_bder_dict['ser']['ref'] = dict()
    file_bder_dict['ser']['hyp'] = dict()

    file_bder_dict['jer'] = dict()
    file_bder_dict['jer']['ref'] = dict()
    file_bder_dict['jer']['hyp'] = dict()

    ref = file_rttm_dict['ref']
    if 'hyp' not in file_rttm_dict.keys(): # no predictions for current file
        hyp = []
    else:
        hyp = file_rttm_dict['hyp']

    ref_seg_len = len(ref)
    hyp_seg_len = len(hyp)
    ref_dict, hyp_dict = dict(), dict()
    ref_set, hyp_set = set(), set()

    for spk_name, start, end in ref:
        if spk_name not in ref_dict.keys():
            ref_dict[spk_name] = []
        ref_dict[spk_name].append((start, end))
        ref_set.add(spk_name)

    for spk_name, start, end in hyp:
        if spk_name not in hyp_dict.keys():
            hyp_dict[spk_name] = []
        hyp_dict[spk_name].append((start, end))
        hyp_set.add(spk_name)

    for spk_name in ref_set:
        # init for segment-level 
        file_bder_dict['ser']['ref'][spk_name] = dict()
        file_bder_dict['ser']['ref'][spk_name]['ms_num'] = 0
        file_bder_dict['ser']['ref'][spk_name]['total_num'] = len(ref_dict[spk_name])
        file_bder_dict['ser']['ref'][spk_name]['ms_case'] = []
        file_bder_dict['ser']['ref'][spk_name]['ms_reason'] = []
        file_bder_dict['ser']['ref'][spk_name]['iou_threshold'] = []

        # init for duration-level
        file_bder_dict['jer']['ref'][spk_name] = dict()
        file_bder_dict['jer']['ref'][spk_name]['fa_duration'] = 0
        file_bder_dict['jer']['ref'][spk_name]['ms_duration'] = 0
        file_bder_dict['jer']['ref'][spk_name]['ref_duration'] = sum([ end - start for start,end in ref_dict[spk_name]])
        file_bder_dict['jer']['ref'][spk_name]['union_duration'] = 0

    ref_index, hyp_index, index_ref, index_hyp = build_speaker_list(ref, hyp)
    cost_matrix = build_cost_matrix(ref, hyp, ref_index, hyp_index)
    row_index, col_index = optimize.linear_sum_assignment(-cost_matrix) # optimal mapping

    matched_ref_set, matched_hyp_set = set(), set()
    for ref_id, hyp_id in zip(row_index, col_index): # matched result
        matched_ref_set.add(index_ref[ref_id])
        matched_hyp_set.add(index_hyp[hyp_id])
        file_bder_dict['optimal_mapping']['r2h'][index_ref[ref_id]] = index_hyp[hyp_id]
        file_bder_dict['optimal_mapping']['h2r'][index_hyp[hyp_id]] = index_ref[ref_id]

    optimal_matching_failed_seg_num = 0

    notmatched_ref_set = ref_set - matched_ref_set # unmatched ref result (missed reference speakers)
    for ref_name in notmatched_ref_set:
        optimal_matching_failed_seg_num += len(ref_dict[ref_name])

        file_bder_dict['ser']['ref'][ref_name]['ms_num'] = len(ref_dict[ref_name])
        file_bder_dict['ser']['ref'][ref_name]['total_num'] = len(ref_dict[ref_name])

        file_bder_dict['jer']['ref'][ref_name] = dict()
        file_bder_dict['jer']['ref'][ref_name]['fa_duration'] = 0
        file_bder_dict['jer']['ref'][ref_name]['ms_duration'] = 0
        file_bder_dict['jer']['ref'][ref_name]['ref_duration'] = 0
        file_bder_dict['jer']['ref'][ref_name]['union_duration'] = 0

        for start, end in ref_dict[ref_name]:

            # all segments are regarded as errors
            file_bder_dict['optimal_mapping']['r_ms_case'].append((start, end))
            file_bder_dict['ser']['ref'][ref_name]['ms_case'].append((start, end))
            file_bder_dict['ser']['ref'][ref_name]['ms_reason'].append('optimalmapping')
            file_bder_dict['ser']['ref'][ref_name]['iou_threshold'].append(['na'])

            # all segments are regarded as missed speech
            file_bder_dict['jer']['ref'][ref_name]['ms_duration'] += end-start
            file_bder_dict['jer']['ref'][ref_name]['ref_duration'] += end-start
            file_bder_dict['jer']['ref'][ref_name]['union_duration'] += end-start

            detailed_result_list.append("%s %f %f %s %s %s\n"%(filename, start, end, ref_name, 'optimalmapping', "ref"))


        file_bder_dict['optimal_mapping']['r_ms_id'].add(ref_name)


    unmatched_hyp_set = hyp_set - matched_hyp_set
    for hyp_name in unmatched_hyp_set: # unmatched ref result (false alarm speakers)
                                       # errors in this part are regarded as speaker^fa part in BER
        optimal_matching_failed_seg_num += len(hyp_dict[hyp_name])

        file_bder_dict['ser']['hyp'][hyp_name] = dict()
        file_bder_dict['ser']['hyp'][hyp_name]['ms_num'] = len(hyp_dict[hyp_name])
        file_bder_dict['ser']['hyp'][hyp_name]['total_num'] = len(hyp_dict[hyp_name])
        file_bder_dict['ser']['hyp'][hyp_name]['ms_case'] = []
        file_bder_dict['ser']['hyp'][hyp_name]['ms_reason'] = []
        file_bder_dict['ser']['hyp'][hyp_name]['iou_threshold'] = []

        file_bder_dict['jer']['hyp'][hyp_name] = dict()
        file_bder_dict['jer']['hyp'][hyp_name]['fa_duration'] = 0
        file_bder_dict['jer']['hyp'][hyp_name]['ms_duration'] = 0
        file_bder_dict['jer']['hyp'][hyp_name]['ref_duration'] = 0
        file_bder_dict['jer']['hyp'][hyp_name]['union_duration'] = 0


        for start, end in hyp_dict[hyp_name]:
            # all segments are regarded as errors
            file_bder_dict['optimal_mapping']['h_ms_case'].append((start, end))

            file_bder_dict['ser']['hyp'][hyp_name]['ms_case'].append((start, end))
            file_bder_dict['ser']['hyp'][hyp_name]['ms_reason'].append('optimalmapping')
            file_bder_dict['ser']['hyp'][hyp_name]['iou_threshold'].append(['na'])

            # all segments are regarded as missed speech
            file_bder_dict['jer']['hyp'][hyp_name]['ms_duration'] += end-start
            file_bder_dict['jer']['hyp'][hyp_name]['ref_duration'] += end-start
            file_bder_dict['jer']['hyp'][hyp_name]['union_duration'] += end-start

            detailed_result_list.append("%s %f %f %s %s %s\n"%(filename, start, end, hyp_name, 'optimalmapping', "hyp"))

        file_bder_dict['optimal_mapping']['h_ms_id'].add(hyp_name)

    # init for matched speakers
    for spk_name in matched_hyp_set:
        file_bder_dict['ser']['hyp'][spk_name] = dict()
        file_bder_dict['ser']['hyp'][spk_name]['ms_num'] = 0
        file_bder_dict['ser']['hyp'][spk_name]['total_num'] = len(hyp_dict[spk_name])
        file_bder_dict['ser']['hyp'][spk_name]['ms_case'] = []
        file_bder_dict['ser']['hyp'][spk_name]['ms_reason'] = []
        file_bder_dict['ser']['hyp'][spk_name]['iou_threshold'] = []


    # core part for SER and BER
    unmatched_segs_iou = 0
    unmatched_segs_missing_ref = 0
    unmatched_segs_missing_hypo = 0

    for ref_id, hyp_id in zip(row_index, col_index):

        ref_segments = ref_dict[index_ref[ref_id]]
        ref_segments.sort()
        hyp_segments = hyp_dict[index_hyp[hyp_id]]
        hyp_segments.sort()

        ref_name = index_ref[ref_id]
        hyp_name = index_hyp[hyp_id]

        # duration-level errors
        file_bder_dict['jer']['ref'][ref_name] = dict()
        fa_duration, ms_duration, intersection_duration, ref_duration, union_duration = get_fa_ms(ref_segments, hyp_segments)
        file_bder_dict['jer']['ref'][ref_name]['fa_duration'] = fa_duration
        file_bder_dict['jer']['ref'][ref_name]['ms_duration'] = ms_duration
        file_bder_dict['jer']['ref'][ref_name]['union_duration'] = union_duration
        file_bder_dict['jer']['ref'][ref_name]['ref_duration'] = ref_duration

        # segment-level errors
        connection_matrix = build_connection_matrix(ref_segments,hyp_segments)
        total_pair = get_all_connected_graphs(connection_matrix)
        total_ref_seg_id_set, total_hyp_seg_id_set = set(list(range(len(connection_matrix)))), set(list(range(len(connection_matrix[0]))))
        total_single_ref_seg_id_set, total_single_hyp_seg_id_set = set(), set()

        for p, single_pair in enumerate(total_pair):
            single_ref_seg_id_set, single_hyp_seg_id_set = set(), set()
            for ref_seg_id, hyp_seg_id in single_pair:
                single_ref_seg_id_set.add(ref_seg_id)
                single_hyp_seg_id_set.add(hyp_seg_id)
                total_single_ref_seg_id_set.add(ref_seg_id)
                total_single_hyp_seg_id_set.add(hyp_seg_id)

            single_ref_segs, single_hyp_segs = [],[]
            for seg_id in single_ref_seg_id_set:
                single_ref_segs.append(ref_segments[seg_id])
            for seg_id in single_hyp_seg_id_set:
                single_hyp_segs.append(hyp_segments[seg_id])


            iou, ref_len, hyp_len, intersection, union = iou_of_pair_segments(single_ref_segs, single_hyp_segs)

            iou_adapt = get_dynamic_iou(single_ref_segs, single_hyp_segs)

            if iou < iou_adapt:
                unmatched_segs_iou += len(single_ref_seg_id_set) # only reference segs are considered as errors
                                                                 # this can avoid the effects of arbitary segmentation of hypothesis

                for start, end in single_ref_segs:

                    file_bder_dict['ser']['ref'][ref_name]['ms_num'] += 1 # marked as ref missed error
                    file_bder_dict['ser']['ref'][ref_name]['ms_case'].append((start,end))
                    file_bder_dict['ser']['ref'][ref_name]['ms_reason'].append('low_iou')
                    file_bder_dict['ser']['ref'][ref_name]['iou_threshold'].append(['%0.3f(%0.3f)'%(iou,iou_adapt)])

                    detailed_result_list.append("%s %f %f %s %s %s\n"%(filename, start, end, ref_name, 'N_%d_%0.2f'%(p,iou_adapt), "ref"))


                for start, end in single_hyp_segs:
                    file_bder_dict['ser']['hyp'][hyp_name]['ms_num'] += 1 # marked as hyp missed error, not used in our metric
                    file_bder_dict['ser']['hyp'][hyp_name]['ms_case'].append((start,end))
                    file_bder_dict['ser']['hyp'][hyp_name]['ms_reason'].append('low_iou')
                    file_bder_dict['ser']['hyp'][hyp_name]['iou_threshold'].append(['%0.3f(%0.3f)'%(iou,iou_adapt)])

                    detailed_result_list.append("%s %f %f %s %s %s\n"%(filename, start, end, ref_name, 'N_%d_%0.2f'%(p,iou_adapt), "hyp"))

            else:
                # save IoU matched result
                for start, end in single_ref_segs:
                    detailed_result_list.append("%s %f %f %s %s %s\n"%(filename, start, end, ref_name, 'Y_%d_%0.2f'%(p,iou_adapt), "ref"))
                for start, end in single_hyp_segs:
                    detailed_result_list.append("%s %f %f %s %s %s\n"%(filename, start, end, ref_name, 'Y_%d_%0.2f'%(p,iou_adapt), "hyp"))

        # for isolated nodes 
        unmatched_segs_ms_ref_set = total_ref_seg_id_set - total_single_ref_seg_id_set
        unmatched_segs_ms_hypo_set = total_hyp_seg_id_set - total_single_hyp_seg_id_set

        unmatched_segs_missing_ref += len(unmatched_segs_ms_ref_set)
        unmatched_segs_missing_hypo += len(unmatched_segs_ms_hypo_set)


        for seg_id in unmatched_segs_ms_ref_set:
            ref_name = index_ref[ref_id]
            start, end = ref_segments[seg_id]
            file_bder_dict['ser']['ref'][ref_name]['ms_num'] += 1
            file_bder_dict['ser']['ref'][ref_name]['ms_case'].append((start,end))
            file_bder_dict['ser']['ref'][ref_name]['ms_reason'].append('isolated_node')
            file_bder_dict['ser']['ref'][ref_name]['iou_threshold'].append(['na'])

            detailed_result_list.append("%s %f %f %s %s %s\n"%(filename, start, end, ref_name, 'alone', "ref"))

        for seg_id in unmatched_segs_ms_hypo_set:
            hyp_name = index_hyp[hyp_id]
            start, end = hyp_segments[seg_id]
            file_bder_dict['ser']['hyp'][hyp_name]['ms_num'] += 1
            file_bder_dict['ser']['hyp'][hyp_name]['ms_case'].append((start,end))
            file_bder_dict['ser']['hyp'][hyp_name]['ms_reason'].append('isolated_node')
            file_bder_dict['ser']['hyp'][hyp_name]['iou_threshold'].append(['na'])

            detailed_result_list.append("%s %f %f %s %s %s\n"%(filename, start, end, hyp_name, 'alone', "hyp"))


    unmatched = unmatched_segs_missing_ref + unmatched_segs_missing_hypo + unmatched_segs_iou + optimal_matching_failed_seg_num

    return filename, file_bder_dict, detailed_result_list

def main():
    parser = ArgumentParser(
        description='SD diarization from RTTM files.', add_help=True,
        usage='%(prog)s [options]')
    parser.add_argument('-s', dest='sys_rttm_fns', help='system RTTM files (default: %(default)s)')
    parser.add_argument('-r', dest='ref_rttm_fns', help='reference RTTM files (default: %(default)s)')
    parser.add_argument('-d', dest='detailed_result', help='Detailed result will be saved in this file', default='detailed_results.txt')
    args = parser.parse_args()

    ref_rttm_file = args.ref_rttm_fns
    ref_total_seg_num = len(open(ref_rttm_file).readlines())

    hyp_rttm_file = args.sys_rttm_fns
    hyp_total_seg_num = len(open(hyp_rttm_file).readlines())

    rttm_dict = read_rttm_files(ref_rttm_file, hyp_rttm_file)

    bder_dict = dict() # meta information dict (record all information for details)
    detailed_result_list = []
    with cf.ProcessPoolExecutor() as executor:
        for filename, file_bder_dict, file_detailed_result_list in executor.map(get_statistics_for_each_file,
                                                                                rttm_dict.items()):
            detailed_result_list.extend(file_detailed_result_list)
            bder_dict[filename] = file_bder_dict

    with open(args.detailed_result, 'w') as f:
        f.writelines(detailed_result_list)

    all_spk_jer, all_spk_ber = [], []   # jaccard error rate(jer) and balanced error rate(ber)
    ser_error_total, total_fa_time, total_ref_time, total_fa_seg_num, total_ref_seg_num = 0, 0, 0, 0, 0

    for filename in bder_dict.keys():

        spk_ser_dict, spk_der_dict = dict(), dict() # segment error rate(ser) and duration error rate(der)
        spk_der = []

        for spk_name in bder_dict[filename]['ser']['ref'].keys():
            ms_num = bder_dict[filename]['ser']['ref'][spk_name]['ms_num']
            total_num = bder_dict[filename]['ser']['ref'][spk_name]['total_num']
            spk_ser_dict[spk_name] = ms_num/total_num
            total_ref_seg_num += total_num
            ser_error_total += ms_num

        for spk_name in bder_dict[filename]['ser']['hyp'].keys():
            ms_reason = bder_dict[filename]['ser']['hyp'][spk_name]['ms_reason']
            ms_case = bder_dict[filename]['ser']['hyp'][spk_name]['ms_case']
            if len(ms_reason) > 0 and ms_reason[0] == 'optimalmapping':
                total_fa_seg_num += len(ms_case)

        for spk_name in bder_dict[filename]['jer']['ref'].keys():
            fa = bder_dict[filename]['jer']['ref'][spk_name]['fa_duration']
            ms = bder_dict[filename]['jer']['ref'][spk_name]['ms_duration']
            ref = bder_dict[filename]['jer']['ref'][spk_name]['ref_duration']
            union = bder_dict[filename]['jer']['ref'][spk_name]['union_duration']

            spk_der_dict[spk_name] = (fa+ms)/ref
            all_spk_jer.append((fa+ms)/union)
            total_ref_time += ref

        for spk_name in bder_dict[filename]['jer']['hyp'].keys():
            fa = bder_dict[filename]['jer']['hyp'][spk_name]['fa_duration']
            ms = bder_dict[filename]['jer']['hyp'][spk_name]['ms_duration']
            ref = bder_dict[filename]['jer']['hyp'][spk_name]['ref_duration']
            total_fa_time += ms

        assert spk_ser_dict.keys() == spk_der_dict.keys()

        for spk_name in spk_ser_dict.keys():
            ser = spk_ser_dict[spk_name]
            der = spk_der_dict[spk_name]

            single_spk_ber = harmonic_mean(ser,der)
            all_spk_ber.append(single_spk_ber)

    jer = sum(all_spk_jer) / len(all_spk_jer)
    ser = ser_error_total / total_ref_seg_num

    ref_spk_ber = sum(all_spk_ber) / len(all_spk_ber)
    fa_duraion = total_fa_time / total_ref_time
    fa_segnum = total_fa_seg_num / total_ref_seg_num
    fa_mean = harmonic_mean(fa_duraion, fa_segnum)
    ber = ref_spk_ber + fa_mean

    return jer, ser, ref_spk_ber, fa_duraion, fa_segnum, fa_mean, ber

if __name__ == "__main__":
   jer, ser, ref_spk_ber, fa_duraion, fa_segnum, fa_mean, ber = main()
   col_names = ['SER', 'BER','ref_part', 'fa_dur', 'fa_seg', 'fa_mean']
   rows = []
   rows.append((ser, ber, ref_spk_ber, fa_duraion, fa_segnum, fa_mean))
   tbl = tabulate(rows, headers=col_names, floatfmt = '.4f', tablefmt='simple')
   print(tbl)
