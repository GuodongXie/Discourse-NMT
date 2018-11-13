# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import operator

import tensorflow as tf


def batch_examples(example, batch_size, max_length, mantissa_bits,
                   shard_multiplier=1, length_multiplier=1, constant=False,
                   num_threads=4, drop_long_sequences=True):
    """ Batch examples

    :param example: A dictionary of <feature name, Tensor>.
    :param batch_size: The number of tokens or sentences in a batch
    :param max_length: The maximum length of a example to keep
    :param mantissa_bits: An integer
    :param shard_multiplier: an integer increasing the batch_size to suit
        splitting across data shards.
    :param length_multiplier: an integer multiplier that is used to
        increase the batch sizes and sequence length tolerance.
    :param constant: Whether to use constant batch size
    :param num_threads: Number of threads
    :param drop_long_sequences: Whether to drop long sequences

    :returns: A dictionary of batched examples
    """

    with tf.name_scope("batch_examples"):
        max_length = max_length or batch_size
        min_length = 8
        mantissa_bits = mantissa_bits

        # Compute boundaries
        x = min_length
        boundaries = []

        while x < max_length:
            boundaries.append(x)
            x += 2 ** max(0, int(math.log(x, 2)) - mantissa_bits)

        # Whether the batch size is constant
        if not constant:
            batch_sizes = [max(1, batch_size // length)
                           for length in boundaries + [max_length]]
            batch_sizes = [b * shard_multiplier for b in batch_sizes]
            bucket_capacities = [2 * b for b in batch_sizes]
        else:
            batch_sizes = batch_size * shard_multiplier
            bucket_capacities = [2 * n for n in boundaries + [max_length]]

        max_length *= length_multiplier
        boundaries = [boundary * length_multiplier for boundary in boundaries]
        max_length = max_length if drop_long_sequences else 10 ** 9

        # The queue to bucket on will be chosen based on maximum length
        max_example_length = 0
        for v in example.values():
            if v.shape.ndims > 0:
                seq_length = tf.shape(v)[0]
                max_example_length = tf.maximum(max_example_length, seq_length)

        (_, outputs) = tf.contrib.training.bucket_by_sequence_length(
            max_example_length,
            example,
            batch_sizes,
            [b + 1 for b in boundaries],
            num_threads=num_threads,
            capacity=2,  # Number of full batches to store, we don't need many.
            bucket_capacities=bucket_capacities,
            dynamic_pad=True,
            keep_input=(max_example_length <= max_length)
        )

    return outputs

def get_training_input_src_context_eager(filenames, params):
    """ Get input for training stage

    :param filenames: A list contains [source_filename, target_filename]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """
    def split_context(src, tgt):
        st = tf.strings.split([src], '####')
        cache_src = st.values[:-1]
        src = st.values[-1]
        return (src, tgt, cache_src)

    def split_sens(src, tgt, context_sens):
        src = tf.string_split([src]).values
        tgt = tf.string_split([tgt]).values
        st = tf.string_split(context_sens)
        context = tf.sparse_to_dense(st.indices, st.dense_shape, st.values, default_value=params.pad)
        eos_flag = tf.not_equal(context, params.pad)  # 找到 非 pad的位置
        context_len = tf.reduce_sum(tf.cast(eos_flag, dtype=tf.int32), axis=1)
        return (src, tgt, context, context_len)

    with tf.device("/cpu:0"):
        src_dataset = tf.data.TextLineDataset(filenames[0])
        src_dataset = src_dataset.map(
            lambda src: tf.strings.regex_replace(src, '####', ' '+params.eos+'####')
        )

        tgt_dataset = tf.data.TextLineDataset(filenames[1])

        dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
        # dataset = dataset.shuffle(params.buffer_size)
        # dataset = dataset.repeat()

        dataset = dataset.map(
            split_context,
            num_parallel_calls=params.num_threads

        )

        # Split string
        dataset = dataset.map(
            split_sens,  # 切分句子，并把context从稀疏转换为标准矩阵，得到context的句子从长度
            num_parallel_calls=params.num_threads
        )

        # dataset = dataset.map(
        #     lambda src, tgt: (
        #         tf.string_split([src]).values,
        #         tf.string_split([tgt]).values,
        #         tf.string_split([src]).values,
        #         2
        #     ),
        #     num_parallel_calls=params.num_threads
        # )


        # Append <eos> symbol
        dataset = dataset.map(
            lambda src, tgt, context, context_sen_len: (
                tf.concat([src, [tf.constant(params.eos)]], axis=0),
                tf.concat([tgt, [tf.constant(params.eos)]], axis=0),
                context,
                context_sen_len
            ),
            num_parallel_calls=params.num_threads
        )

        # Convert to dictionary
        dataset = dataset.map(
            lambda src, tgt, context, context_sen_len: {
                "source": src,
                "target": tgt,
                "source_length": tf.shape(src),
                "target_length": tf.shape(tgt),
                "context": context,
                "context_sen_len": context_sen_len
            },
            num_parallel_calls=params.num_threads
        )
        #
        # # Create iterator
        # iterator = dataset.make_one_shot_iterator()
        # features = iterator.get_next()
        #
        # # Create lookup table
        # src_table = tf.contrib.lookup.index_table_from_tensor(
        #     tf.constant(params.vocabulary["source"]),
        #     default_value=params.mapping["source"][params.unk]
        # )
        # tgt_table = tf.contrib.lookup.index_table_from_tensor(
        #     tf.constant(params.vocabulary["target"]),
        #     default_value=params.mapping["target"][params.unk]
        # )
        #
        # # String to index lookup
        # #features["source_ori"] = features["source"]
        # #features["context_ori"] = features["context"]
        #
        # features["source"] = src_table.lookup(features["source"])
        # features["target"] = tgt_table.lookup(features["target"])
        # features["context"] = src_table.lookup(features["context"])
        #
        # # Batching
        # shard_multiplier = len(params.device_list) * params.update_cycle
        # features = batch_examples(features, params.batch_size,
        #                           params.max_length, params.mantissa_bits,
        #                           shard_multiplier=shard_multiplier,
        #                           length_multiplier=params.length_multiplier,
        #                           constant=params.constant_batch_size,
        #                           num_threads=params.num_threads)
        #
        # # Convert to int32
        # features["source"] = tf.to_int32(features["source"])
        # features["target"] = tf.to_int32(features["target"])
        # features["context"] = tf.to_int32(features["context"])
        # features["context_sen_len"] = tf.to_int32(features["context_sen_len"])
        # features["source_length"] = tf.to_int32(features["source_length"])
        # features["target_length"] = tf.to_int32(features["target_length"])
        # features["source_length"] = tf.squeeze(features["source_length"], 1)
        # features["target_length"] = tf.squeeze(features["target_length"], 1)
        #
        # return features

        return dataset

def get_training_input(filenames, params):
    """ Get input for training stage

    :param filenames: A list contains [source_filename, target_filename]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """

    with tf.device("/cpu:0"):
        src_dataset = tf.data.TextLineDataset(filenames[0])
        tgt_dataset = tf.data.TextLineDataset(filenames[1])

        dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
        dataset = dataset.shuffle(params.buffer_size)
        dataset = dataset.repeat()

        # Split string
        dataset = dataset.map(
            lambda src, tgt: (
                tf.string_split([src]).values,
                tf.string_split([tgt]).values
            ),
            num_parallel_calls=params.num_threads
        )

        # Append <eos> symbol
        dataset = dataset.map(
            lambda src, tgt: (
                tf.concat([src, [tf.constant(params.eos)]], axis=0),
                tf.concat([tgt, [tf.constant(params.eos)]], axis=0)
            ),
            num_parallel_calls=params.num_threads
        )

        # Convert to dictionary
        dataset = dataset.map(
            lambda src, tgt: {
                "source": src,
                "target": tgt,
                "source_length": tf.shape(src),
                "target_length": tf.shape(tgt)
            },
            num_parallel_calls=params.num_threads
        )

        # Create iterator
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Create lookup table
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target"]),
            default_value=params.mapping["target"][params.unk]
        )

        # String to index lookup
        features["source"] = src_table.lookup(features["source"])
        features["target"] = tgt_table.lookup(features["target"])

        # Batching
        shard_multiplier = len(params.device_list) * params.update_cycle
        features = batch_examples(features, params.batch_size,
                                  params.max_length, params.mantissa_bits,
                                  shard_multiplier=shard_multiplier,
                                  length_multiplier=params.length_multiplier,
                                  constant=params.constant_batch_size,
                                  num_threads=params.num_threads)

        # Convert to int32
        features["source"] = tf.to_int32(features["source"])
        features["target"] = tf.to_int32(features["target"])
        features["source_length"] = tf.to_int32(features["source_length"])
        features["target_length"] = tf.to_int32(features["target_length"])
        features["source_length"] = tf.squeeze(features["source_length"], 1)
        features["target_length"] = tf.squeeze(features["target_length"], 1)

        return features


def get_training_input_src_context(filenames, params):
    """ Get input for training stage

    :param filenames: A list contains [source_filename, target_filename]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """
    def split_context(src, tgt):
        st = tf.strings.split([src], '####')
        cache_src = st.values[:-1]
        src = st.values[-1]
        return (src, tgt, cache_src)

    def split_sens(src, tgt, context_sens):
        src = tf.string_split([src]).values
        tgt = tf.string_split([tgt]).values
        st = tf.string_split(context_sens)
        context = tf.sparse_to_dense(st.indices, st.dense_shape, st.values, default_value=params.pad)
        eos_flag = tf.not_equal(context, params.pad)  # 找到 非 pad的位置
        context_len = tf.reduce_sum(tf.cast(eos_flag, dtype=tf.int32), axis=1)
        return (src, tgt, context, context_len)

    with tf.device("/cpu:0"):
        src_dataset = tf.data.TextLineDataset(filenames[0])
        src_dataset = src_dataset.map(
            lambda src: tf.strings.regex_replace(src, '####', ' '+params.eos+'####')
        )

        tgt_dataset = tf.data.TextLineDataset(filenames[1])

        dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
        dataset = dataset.shuffle(params.buffer_size)
        dataset = dataset.repeat()

        dataset = dataset.map(
            split_context,
            num_parallel_calls=params.num_threads

        )

        # Split string
        dataset = dataset.map(
            split_sens,  # 切分句子，并把context从稀疏转换为标准矩阵，得到context的句子从长度
            num_parallel_calls=params.num_threads
        )

        # dataset = dataset.map(
        #     lambda src, tgt: (
        #         tf.string_split([src]).values,
        #         tf.string_split([tgt]).values,
        #         tf.string_split([src]).values,
        #         2
        #     ),
        #     num_parallel_calls=params.num_threads
        # )


        # Append <eos> symbol
        dataset = dataset.map(
            lambda src, tgt, context, context_sen_len: (
                tf.concat([src, [tf.constant(params.eos)]], axis=0),
                tf.concat([tgt, [tf.constant(params.eos)]], axis=0),
                context,
                context_sen_len
            ),
            num_parallel_calls=params.num_threads
        )

        # Convert to dictionary
        dataset = dataset.map(
            lambda src, tgt, context, context_sen_len: {
                "source": src,
                "target": tgt,
                "source_length": tf.shape(src),
                "target_length": tf.shape(tgt),
                "context": context,
                "context_sen_len": context_sen_len
            },
            num_parallel_calls=params.num_threads
        )

        # Create iterator
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Create lookup table
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target"]),
            default_value=params.mapping["target"][params.unk]
        )

        # String to index lookup
        #features["source_ori"] = features["source"]
        #features["context_ori"] = features["context"]

        features["source"] = src_table.lookup(features["source"])
        features["target"] = tgt_table.lookup(features["target"])
        features["context"] = src_table.lookup(features["context"])

        # Batching
        shard_multiplier = len(params.device_list) * params.update_cycle
        features = batch_examples(features, params.batch_size,
                                  params.max_length, params.mantissa_bits,
                                  shard_multiplier=shard_multiplier,
                                  length_multiplier=params.length_multiplier,
                                  constant=params.constant_batch_size,
                                  num_threads=params.num_threads)

        # Convert to int32
        features["source"] = tf.to_int32(features["source"])
        features["target"] = tf.to_int32(features["target"])
        features["context"] = tf.to_int32(features["context"])
        features["context_sen_len"] = tf.to_int32(features["context_sen_len"])
        features["source_length"] = tf.to_int32(features["source_length"])
        features["target_length"] = tf.to_int32(features["target_length"])
        features["source_length"] = tf.squeeze(features["source_length"], 1)
        features["target_length"] = tf.squeeze(features["target_length"], 1)

        return features


def sort_input_file_catch(filename, reverse=True):

    def get_real_src_len(input_sen):
        src_list = input_sen.split('####')
        return len(src_list[-1].split())  # 仅仅返回当前句的长度

    # Read file
    with tf.gfile.Open(filename) as fd:
        inputs = [line.strip() for line in fd]

    input_lens = []
    for i, line in enumerate(inputs):
        input_lens.append((i, get_real_src_len(line)))

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=reverse)
    sorted_keys = {}
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])
        sorted_keys[index] = i

    return sorted_keys, sorted_inputs

def sort_and_zip_files(names):

    def get_real_src_len(input_sen):
        src_list = input_sen.split('####')
        return len(src_list[-1])


    inputs = []
    input_lens = []
    files = [tf.gfile.GFile(name) for name in names]

    count = 0

    for lines in zip(*files):
        lines = [line.strip() for line in lines]
        input_lens.append((count, get_real_src_len(lines[0])))
        inputs.append(lines)
        count += 1

    # Close files
    for fd in files:
        fd.close()

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=True)
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])

    return [list(x) for x in zip(*sorted_inputs)]


def get_evaluation_input_catch(inputs, params):

    def split_context(src):
        st = tf.strings.split([src], '####')
        cache_src = st.values[:-1]
        src = st.values[-1]
        return (src, cache_src)

    def split_sens(src, context_sens):
        src = tf.string_split([src]).values
        st = tf.string_split(context_sens)
        context = tf.sparse_to_dense(st.indices, st.dense_shape, st.values, default_value=params.pad)
        eos_flag = tf.not_equal(context, params.pad)  # 找到 非 pad的位置
        context_len = tf.reduce_sum(tf.cast(eos_flag, dtype=tf.int32), axis=1)
        return (src, context, context_len)
    with tf.device("/cpu:0"):
        # Create datasets
        datasets = []

        #  首先处理src+context
        dataset_src = tf.data.Dataset.from_tensor_slices(inputs[0]) # source with context

        dataset_src = dataset_src.map(
            lambda src: tf.strings.regex_replace(src, '####', ' ' + params.eos + '####')
        )
        dataset_src = dataset_src.map(
            lambda src: tf.strings.join([src, params.eos], ' ')
        )

        dataset_src = dataset_src.map(
            split_context,
            num_parallel_calls=params.num_threads

        )
        dataset_src = dataset_src.map(
            split_sens,
            num_parallel_calls=params.num_threads

        )
        datasets.append(dataset_src)

        # start process reference
        for data in inputs[1:]:
            dataset = tf.data.Dataset.from_tensor_slices(data)
            # Split string
            dataset = dataset.map(lambda x: tf.string_split([x]).values,
                                  num_parallel_calls=params.num_threads)
            # Append <eos>
            dataset = dataset.map(
                lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
                num_parallel_calls=params.num_threads
            )
            datasets.append(dataset)
        dataset = tf.data.Dataset.zip(tuple(datasets))

        # Convert tuple to dictionary

        dataset = dataset.map(
            lambda *x: {
                "source": x[0][0],
                "source_length": tf.shape(x[0][0])[0],
                "context": x[0][1],
                "context_sen_len": x[0][2],
                "references": x[1:]
            },
            num_parallel_calls=params.num_threads
        )

        # 此时还没有batch, 是输入一个句子和它的context
        dataset = dataset.padded_batch(
            params.eval_batch_size,
            {
                "source": [tf.Dimension(None)],
                "source_length": [],
                "context": [tf.Dimension(None), tf.Dimension(None)],
                "context_sen_len": [tf.Dimension(None)],
                "references": (tf.Dimension(None),) * (len(inputs) - 1)
            },
            {
                "source": params.pad,
                "source_length": 0,
                "context": params.pad,
                "context_sen_len": 0,
                "references": (params.pad,) * (len(inputs) - 1)
            }
        )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Covert source symbols to ids
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )

        features["source"] = src_table.lookup(features["source"])
        features["context"] = src_table.lookup(features["context"])

    return features


def get_inference_input(inputs, params):

    def split_context(src):
        st = tf.strings.split([src], '####')
        cache_src = st.values[:-1]
        src = st.values[-1]
        return (src, cache_src)

    def split_sens(src, context_sens):
        src = tf.string_split([src]).values
        st = tf.string_split(context_sens)
        context = tf.sparse_to_dense(st.indices, st.dense_shape, st.values, default_value=params.pad)
        eos_flag = tf.not_equal(context, params.pad)  # 找到 非 pad的位置
        context_len = tf.reduce_sum(tf.cast(eos_flag, dtype=tf.int32), axis=1)
        return (src, context, context_len)

    with tf.device("/cpu:0"):
        dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(inputs)
        )

        dataset = dataset.map(
            lambda src: tf.strings.regex_replace(src, '####', ' ' + params.eos + '####')
        )

        dataset = dataset.map(
            split_context,
            num_parallel_calls=params.num_threads

        )

        # Split each sentence string
        dataset = dataset.map(
            split_sens,
            num_parallel_calls=params.num_threads)

        # Append <eos>
        dataset = dataset.map(
            lambda x, context, context_len: (tf.concat([x, [tf.constant(params.eos)]], axis=0), context, context_len),
            num_parallel_calls=params.num_threads
        )

        # Convert tuple to dictionary
        dataset = dataset.map(
            lambda x, context, context_len: {"source": x, "source_length": tf.shape(x)[0],
                                             'context': context, 'context_sen_len': context_len},
            num_parallel_calls=params.num_threads
        )

        dataset = dataset.padded_batch(
            params.decode_batch_size * len(params.device_list),
            {
                "source": [tf.Dimension(None)], "source_length": [],
                "context": [tf.Dimension(None), tf.Dimension(None)],
                'context_sen_len': [tf.Dimension(None)]
            },
            {"source": params.pad, "source_length": 0, "context": params.pad, "context_sen_len": 0,}
        )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        features["source_ori"] = features["source"]
        features["context_ori"] = features["context"]
        features["source"] = src_table.lookup(features["source"])
        features["context"] = src_table.lookup(features["context"])

        return features
