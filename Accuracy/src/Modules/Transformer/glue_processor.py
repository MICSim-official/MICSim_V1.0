import os

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    
    def get_train_examples(self, data_dir):
      """Gets a collection of `InputExample`s for the train set."""
      raise NotImplementedError()   
    
    def get_dev_examples(self, data_dir):
      """Gets a collection of `InputExample`s for the dev set."""
      raise NotImplementedError()   
    
    def get_test_examples(self, data_dir):
      """Gets a collection of `InputExample`s for prediction."""
      raise NotImplementedError()   
    
    def get_labels(self):
      """Gets the list of labels for this data set."""
      raise NotImplementedError()   
    
    @classmethod
    def _read_tsv(cls, input_file):
      """Reads a tab separated value file."""
      data = []
      with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("\t")
            data.append(line)
      return data


class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return [0, 1]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    label = 0
    for (i, line) in enumerate(lines):
        # Only the test set has a header
        if set_type == "test" and i == 0:
            continue
        if len(line) != 2:
            label = int(line[-3])
        examples.append([line[-1], label])
    return examples

class MnliProcessor(DataProcessor):
  """Processor for the MNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
        "dev")

  def get_dev_mismatched_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
        "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

  def get_test_mismatched_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    label = "contradiction"
    all_labels = ["contradiction", "entailment", "neutral"]
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        if set_type == "train":
            text1, text2, label = line[-4], line[-3], line[-1]
        elif set_type == "dev":
            text1, text2, label = line[-8], line[-7], line[-1]
        else:
            text1, text2 = line[-2], line[-1]
        examples.append([text1, text2, all_labels.index(label)])
    return examples

class QnliProcessor(DataProcessor):
  """Processor for the QNLI and RTE data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["not_entailment", "entailment"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    label = "entailment"
    all_labels = ["not_entailment", "entailment"]
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        if set_type == "test":
            _, text1, text2 = line
        else:
            _, text1, text2, label = line
        examples.append([text1, text2, all_labels.index(label)])
    return examples

class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "msr_paraphrase_train.txt")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "msr_paraphrase_test.txt")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return [0, 1]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    label = 0
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        if set_type == "test":
            text1, text2 = line[-2:]
        else:
            text1, text2, label = line[-2], line[-1], int(line[0])
        examples.append([text1, text2, label])
    return examples

class QqpProcessor(DataProcessor):
  """Processor for the QQP and WNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return [0, 1]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    label = 0
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        if set_type == "test":
            _, text1, text2 = line
        else:
            text1, text2, label = line[-3:]
        examples.append([text1, text2, int(label)])
    return examples

class Sst2Processor(DataProcessor):
  """Processor for the SST-2 data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return [0, 1]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    label = 0
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        if set_type == "test":
            text1 = line[-1]
        else:
            text1, label = line
        examples.append([text1, int(label)])
    return examples

class StsbProcessor(DataProcessor):
  """Processor for the STS-B data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return 0.0

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    label = 0.
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        if set_type == "test":
            text1, text2 = line[-2:]
        else:
            text1, text2, label = line[-3:]
        examples.append([text1, text2, float(label)])
    return examples
