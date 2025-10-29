import os
from tags import PHITag
from xml.etree import ElementTree


class Annotation(object):

    def __init__(self, file_name=None, root="root"):
        self.doc_id = ''
        self.sys_id = ''
        self.text = None
        self.num_sentences = None
        self.root = root
        self.sensitive_spans = []
        self.sensitive_spans_merged = []
        self.verbose = False

        if file_name:
            self.sys_id = os.path.basename(os.path.dirname(file_name))
            self.doc_id = os.path.splitext(os.path.basename(file_name))[0]
        else:
            self.doc_id = None

    @property
    def id(self):
        return self.doc_id

    def get_phi(self):
        return self.phi

    def get_phi_spans(self):
        return self.sensitive_spans

    def get_phi_spans_merged(self):
        return self.sensitive_spans_merged

    def get_phi_spans_joins(self):
        return self.sensitive_spans_joins

    def get_number_sentences(self):
        try:
            self.num_sentences = \
                sum(1 for line in open('annotated_corpora/sentence_splitted/' +
                                       self.doc_id +
                                       ".ann"))
        except IOError:
            print("File '" +
                  'freeling/sentence_splitted/' +
                  self.doc_id +
                  ".ann' not found.")
        return self.num_sentences

    def add_spans(self, phi_tags):
        for tag in sorted(phi_tags):
            self.sensitive_spans.append(tag)

        for y in sorted(phi_tags):
            if not self.sensitive_spans_merged:
                self.sensitive_spans_merged.append(y)
            else:
                x = self.sensitive_spans_merged.pop()
                if self.is_all_non_alphanumeric(self.text[x[1]:y[0]]):
                    self.sensitive_spans_merged.append((x[0], y[1]))
                else:
                    self.sensitive_spans_merged.append(x)
                    self.sensitive_spans_merged.append(y)

    @staticmethod
    def is_all_non_alphanumeric(string):
        for i in string:
            if i.isalnum():
                return False
        return True


class i2b2Annotation(Annotation):
    """ This class models the i2b2 annotation format."""

    def __init__(self, file_name=None, root="root"):
        self.doc_id = ''
        self.sys_id = ''
        self.text = None
        self.num_sentences = None
        self.root = root
        self.phi = []
        self.sensitive_spans = []
        self.sensitive_spans_merged = []
        self.verbose = False

        if file_name:
            self.sys_id = os.path.basename(os.path.dirname(file_name))
            self.doc_id = os.path.splitext(os.path.basename(file_name))[0]

            self.parse_text_and_tags(file_name)
            self.parse_text_and_spans(file_name)
            self.file_name = file_name
        else:
            self.doc_id = None

    def parse_text_and_tags(self, file_name=None):
        if file_name is not None:
            text = open(file_name, 'r').read()
            self.text = text

            tree = ElementTree.parse(file_name)
            root = tree.getroot()

            self.root = root.tag

            try:
                self.text = root.find("TEXT").text
            except AttributeError:
                self.text = None

            # Handles files where PHI, and AnnotatorTags are all just
            # stuffed into tag element.
            for t, cls in PHITag.tag_types.items():
                if len(root.find("TAGS").findall(t)):
                    for element in root.find("TAGS").findall(t):
                        self.phi.append(cls(element))

    def parse_text_and_spans(self, file_name=None):

        if file_name is not None:
            text = open(file_name, 'r').read()
            self.text = text

            tree = ElementTree.parse(file_name)
            root = tree.getroot()

            self.root = root.tag

            try:
                self.text = root.find("TEXT").text
            except AttributeError:
                self.text = None

            # Fill list with tuples (start, end) for each annotation
            phi_tags = []
            for t, cls in PHITag.tag_types.items():
                if len(root.find("TAGS").findall(t)):
                    for element in root.find("TAGS").findall(t):
                        phi_tags.append((cls(element).get_start(), cls(element).get_end()))

            # Store spans
            self.add_spans(phi_tags)


class BratAnnotation(Annotation):
    """ This class models the BRAT annotation format."""

    def __init__(self, file_name=None, root="root"):
        self.doc_id = ''
        self.sys_id = ''
        self.text = None
        self.num_sentences = None
        self.root = root
        self.phi = []
        self.sensitive_spans = []
        self.sensitive_spans_merged = []
        self.verbose = False

        if file_name:
            self.sys_id = os.path.basename(os.path.dirname(file_name))
            self.doc_id = os.path.splitext(os.path.basename(file_name))[0]

            self.parse_text_and_tags(file_name)
            self.parse_text_and_spans(file_name)
            self.file_name = file_name
        else:
            self.doc_id = None

    def parse_text_and_tags(self, file_name=None):
        if file_name is not None:
            text = open(os.path.splitext(file_name)[0] + '.txt', 'r').read()
            self.text = text

            for row in open(file_name, 'r'):
                line = row.strip()
                if line.startswith("T"):  # Lines is a Brat TAG
                    try:
                        label = line.split("\t")[1].split()
                        tag = label[0]
                        start = int(label[1])
                        end = int(label[2])
                        self.phi.append((tag, start, end))
                    except IndexError:
                        print("ERROR! Index error while splitting sentence '" +
                              line + "' in document '" + file_name + "'!")
                else:  # Line is a Brat comment
                    if self.verbose:
                        print("\tSkipping line (comment):\t" + line)

    def parse_text_and_spans(self, file_name=None):

        if file_name is not None:
            text = open(os.path.splitext(file_name)[0] + '.txt', 'r').read()
            self.text = text

            phi_tags = []
            for row in open(file_name, 'r'):
                line = row.strip()
                if line.startswith("T"):  # Lines is a Brat TAG
                    try:
                        label = line.split("\t")[1].split()
                        start = int(label[1])
                        end = int(label[2])

                        phi_tags.append((start, end))
                    except IndexError:
                        print("ERROR! Index error while splitting sentence '" +
                              line + "' in document '" + file_name + "'!")
                else:  # Line is a Brat comment
                    if self.verbose:
                        print("\tSkipping line (comment):\t" + line)

            # Store spans
            self.add_spans(phi_tags)


class Evaluate(object):
    """Base class with all methods to evaluate the different subtracks."""

    def __init__(self, sys_ann, gs_ann):
        self.tp = []
        self.fp = []
        self.fn = []
        self.doc_ids = []
        self.verbose = False

        self.sys_id = sys_ann[list(sys_ann.keys())[0]].sys_id

    @staticmethod
    def get_tagset_ner(annotation):
        return annotation.get_phi()

    @staticmethod
    def get_tagset_span(annotation):
        return annotation.get_phi_spans()

    @staticmethod
    def get_tagset_span_merged(annotation):
        return annotation.get_phi_spans_merged()

    @staticmethod
    def is_contained(content, container):
        for element in sorted(container):
            if content[0] >= element[0] and content[1] <= element[1]:
                return True
        return False

    @staticmethod
    def recall(tp, fn):
        try:
            return len(tp) / float(len(fn) + len(tp))
        except ZeroDivisionError:
            return 0.0

    @staticmethod
    def precision(tp, fp):
        try:
            return len(tp) / float(len(fp) + len(tp))
        except ZeroDivisionError:
            return 0.0

    @staticmethod
    def F_beta(p, r, beta=1):
        try:
            return (1 + beta**2) * ((p * r) / (p + r))
        except ZeroDivisionError:
            return 0.0

    def micro_recall(self):
        try:
            return sum([len(t) for t in self.tp]) /  \
                float(sum([len(t) for t in self.tp]) +
                      sum([len(t) for t in self.fn]))
        except ZeroDivisionError:
            return 0.0

    def micro_precision(self):
        try:
            return sum([len(t) for t in self.tp]) /  \
                float(sum([len(t) for t in self.tp]) +
                      sum([len(t) for t in self.fp]))
        except ZeroDivisionError:
            return 0.0

    def _print_docs(self):
        for i, doc_id in enumerate(self.doc_ids):
            mp = Evaluate.precision(self.tp[i], self.fp[i])
            mr = Evaluate.recall(self.tp[i], self.fn[i])

            str_fmt = "{:<35}{:<15}{:<20}"

            print(str_fmt.format(doc_id,
                                 "Precision",
                                 "{:.4}".format(mp)))

            print(str_fmt.format("",
                                 "Recall",
                                 "{:.4}".format(mr)))

            print(str_fmt.format("",
                                 "F1",
                                 "{:.4}".format(Evaluate.F_beta(mp, mr))))

            print("{:-<60}".format(""))

    def _print_summary(self):
        mp = self.micro_precision()
        mr = self.micro_recall()

        str_fmt = "{:<35}{:<15}{:<20}"

        print(str_fmt.format("", "", ""))

        print("Report (" + self.sys_id + "):")

        print("{:-<60}".format(""))

        print(str_fmt.format(self.label,
                             "Measure", "Micro"))

        print("{:-<60}".format(""))

        print(str_fmt.format("Total ({} docs)".format(len(self.doc_ids)),
                             "Precision",
                             "{:.4}".format(mp)))

        print(str_fmt.format("",
                             "Recall",
                             "{:.4}".format(mr)))

        print(str_fmt.format("",
                             "F1",
                             "{:.4}".format(Evaluate.F_beta(mr, mp))))

        print("{:-<60}".format(""))

        print("\n")

    def print_docs(self):
        print("\n")
        print("Report ({}):".format(self.sys_id))
        print("{:-<60}".format(""))
        print("{:<35}{:<15}{:<20}".format("Document ID", "Measure", "Micro"))
        print("{:-<60}".format(""))
        self._print_docs()

    def print_report(self, verbose=False):
        self.verbose = verbose
        if verbose:
            self.print_docs()

        self._print_summary()


class EvaluateSubtrack1(Evaluate):
    """Class for running the NER evaluation."""

    def __init__(self, sys_sas, gs_sas):
        self.tp = []
        self.fp = []
        self.fn = []
        self.num_sentences = []
        self.doc_ids = []
        self.verbose = False

        self.sys_id = sys_sas[list(sys_sas.keys())[0]].sys_id
        self.label = "Subtrack 1 [NER]"

        for doc_id in sorted(list(set(sys_sas.keys()) & set(gs_sas.keys()))):

            gold = set(self.get_tagset_ner(gs_sas[doc_id]))
            sys = set(self.get_tagset_ner(sys_sas[doc_id]))
            # num_sentences = self.get_num_sentences(sys_sas[doc_id])

            self.tp.append(gold.intersection(sys))
            self.fp.append(sys - gold)
            self.fn.append(gold - sys)
            # self.num_sentences.append(num_sentences)
            self.doc_ids.append(doc_id)

    @staticmethod
    def get_num_sentences(annotation):
        return annotation.get_number_sentences()

    @staticmethod
    def leak_score(fn, num_sentences):
        try:
            return float(len(fn) / num_sentences)
        except ZeroDivisionError:
            return 0.0
        except TypeError:
            return "NA"

    def micro_leak(self):
        try:
            return float(sum([len(t) for t in self.fn]) / sum(t for t in self.num_sentences))
        except ZeroDivisionError:
            return 0.0
        except TypeError:
            return "NA"

    def _print_docs(self):
        for i, doc_id in enumerate(self.doc_ids):
            mp = EvaluateSubtrack1.precision(self.tp[i], self.fp[i])
            mr = EvaluateSubtrack1.recall(self.tp[i], self.fn[i])
            # leak = EvaluateSubtrack1.leak_score(self.fn[i], self.num_sentences[i])

            str_fmt = "{:<35}{:<15}{:<20}"

            # print(str_fmt.format(doc_id,
            #                      "Leak",
            #                      "{:.4}".format(leak)))

            print(str_fmt.format("",
                                 "Precision",
                                 "{:.4}".format(mp)))

            print(str_fmt.format("",
                                 "Recall",
                                 "{:.4}".format(mr)))

            print(str_fmt.format("",
                                 "F1",
                                 "{:.4}".format(Evaluate.F_beta(mp, mr))))

            print("{:-<60}".format(""))

    def _print_summary(self):
        mp = self.micro_precision()
        mr = self.micro_recall()
        # ml = self.micro_leak()

        str_fmt = "{:<35}{:<15}{:<20}"

        print(str_fmt.format("", "", ""))

        print("Report (" + self.sys_id + "):")

        print("{:-<60}".format(""))

        print(str_fmt.format(self.label,
                             "Measure", "Micro"))

        print("{:-<60}".format(""))

        # print(str_fmt.format("Total ({} docs)".format(len(self.doc_ids)),
        #                      "Leak",
        #                      "{:.4}".format(ml)))

        print(str_fmt.format("Total ({} docs)".format(len(self.doc_ids)),
                             "Precision",
                             "{:.4}".format(mp)))

        print(str_fmt.format("",
                             "Recall",
                             "{:.4}".format(mr)))

        print(str_fmt.format("",
                             "F1",
                             "{:.4}".format(Evaluate.F_beta(mr, mp))))

        print("{:-<60}".format(""))

        print("\n")


class EvaluateSubtrack2(Evaluate):
    """Class for running the SPAN evaluation with strict span mode."""

    def __init__(self, sys_sas, gs_sas):
        self.tp = []
        self.fp = []
        self.fn = []
        self.doc_ids = []
        self.verbose = False

        self.sys_id = sys_sas[list(sys_sas.keys())[0]].sys_id
        self.label = "Subtrack 2 [strict]"

        for doc_id in sorted(list(set(sys_sas.keys()) & set(gs_sas.keys()))):

            gold = set(self.get_tagset_span(gs_sas[doc_id]))
            sys = set(self.get_tagset_span(sys_sas[doc_id]))

            self.tp.append(gold.intersection(sys))
            self.fp.append(sys - gold)
            self.fn.append(gold - sys)
            self.doc_ids.append(doc_id)


class EvaluateSubtrack2merged(Evaluate):
    """Class for running the SPAN evaluation with merged spans mode."""

    def __init__(self, sys_sas, gs_sas):
        self.tp = []
        self.fp = []
        self.fn = []
        self.doc_ids = []
        self.verbose = False

        self.sys_id = sys_sas[list(sys_sas.keys())[0]].sys_id
        self.label = "Subtrack 2 [merged]"

        for doc_id in sorted(list(set(sys_sas.keys()) & set(gs_sas.keys()))):

            gold_strict = set(self.get_tagset_span(gs_sas[doc_id]))
            sys_strict = set(self.get_tagset_span(sys_sas[doc_id]))

            gold_merged = set(self.get_tagset_span_merged(gs_sas[doc_id]))
            sys_merged = set(self.get_tagset_span_merged(sys_sas[doc_id]))

            intersection = gold_strict.intersection(sys_strict).union(gold_merged.intersection(sys_merged))

            fp = sys_strict - gold_strict
            for tag in sys_strict:
                if self.is_contained(tag, intersection):
                    if tag in fp:
                        fp.remove(tag)

            fn = gold_strict - sys_strict
            for tag in gold_strict:
                if self.is_contained(tag, intersection):
                    if tag in fn:
                        fn.remove(tag)

            self.tp.append(intersection)
            self.fp.append(fp)
            self.fn.append(fn)
            self.doc_ids.append(doc_id)


class MeddocanEvaluation(object):
    """Base class for running the evaluations."""

    def __init__(self):
        self.evaluations = []

    def add_eval(self, e, label=""):
        e.sys_id = "SYSTEM: " + e.sys_id
        e.label = label
        self.evaluations.append(e)

    def print_docs(self):
        for e in self.evaluations:
            e.print_docs()

    def print_report(self, verbose=False):
        for e in self.evaluations:
            e.print_report(verbose=verbose)


class NER_Evaluation(MeddocanEvaluation):
    """Class for running the NER evaluation (Subtrack 1)."""

    def __init__(self, annotator_cas, gold_cas, **kwargs):
        self.evaluations = []

        # Basic Evaluation
        self.add_eval(EvaluateSubtrack1(annotator_cas, gold_cas, **kwargs),
                      label="SubTrack 1 [NER]")


class Span_Evaluation(MeddocanEvaluation):
    """Class for running the SPAN evaluation (Subtrack 2). Calls to 'strict'
    and 'merged' evaluations. """

    def __init__(self, annotator_cas, gold_cas, **kwargs):
        self.evaluations = []

        self.add_eval(EvaluateSubtrack2(annotator_cas, gold_cas, **kwargs),
                      label="SubTrack 2 [strict]")

        self.add_eval(EvaluateSubtrack2merged(annotator_cas, gold_cas, **kwargs),
                      label="SubTrack 2 [merged]")

class EntityTypeMetrics:
    """Class to calculate precision, recall, and F1-score grouped by entity type."""

    @staticmethod
    def calculate_metrics(tp, fp, fn):
        """Calculate precision, recall, and F1-score for each entity type."""
        metrics = {}

        # Aggregate true positives, false positives, and false negatives by entity type
        for entity in tp:
            entity_type = entity[0]
            if entity_type not in metrics:
                metrics[entity_type] = {"tp": 0, "fp": 0, "fn": 0}
            metrics[entity_type]["tp"] += 1

        for entity in fp:
            entity_type = entity[0]
            if entity_type not in metrics:
                metrics[entity_type] = {"tp": 0, "fp": 0, "fn": 0}
            metrics[entity_type]["fp"] += 1

        for entity in fn:
            entity_type = entity[0]
            if entity_type not in metrics:
                metrics[entity_type] = {"tp": 0, "fp": 0, "fn": 0}
            metrics[entity_type]["fn"] += 1

        # Calculate precision, recall, and F1-score for each entity type
        for entity_type, counts in metrics.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[entity_type]["precision"] = precision
            metrics[entity_type]["recall"] = recall
            metrics[entity_type]["f1"] = f1

        return metrics


# Example usage:
# metrics = EntityTypeMetrics.calculate_metrics(tp, fp, fn)
# EntityTypeMetrics.print_metrics(metrics)

class TokenLevelMetrics:
    """Class to calculate token-level metrics using BIO tagging."""
    
    @staticmethod
    def create_bio_labels(text, annotations):
        """Create BIO labels for a given text and annotations.
        
        Args:
            text: The text content
            annotations: List of (label, start, end) tuples
            
        Returns:
            List of (token, start, end, bio_label) tuples
        """
        import re
        
        # Tokenize text by whitespace
        tokens = []
        for match in re.finditer(r'\S+', text):
            tokens.append((match.group(), match.start(), match.end()))
        
        # Initialize all tokens as O
        bio_labels = ["O"] * len(tokens)
        
        # Mark entities with BIO tags
        for label, start, end in annotations:
            for i, (token, token_start, token_end) in enumerate(tokens):
                if token_end <= start:
                    continue
                if token_start >= end:
                    break
                if token_start >= start and token_end <= end:
                    if bio_labels[i] == "O":
                        bio_labels[i] = "B-" + label
                    elif bio_labels[i].startswith("B-") or bio_labels[i].startswith("I-"):
                        bio_labels[i] = "I-" + label
        
        return [(token, start, end, bio_label) for (token, start, end), bio_label in zip(tokens, bio_labels)]
    
    @staticmethod
    def calculate_token_metrics(gold_annotations, system_annotations):
        """Calculate token-level metrics using BIO tagging.
        
        Args:
            gold_annotations: Dictionary of gold standard annotations
            system_annotations: Dictionary of system annotations
            
        Returns:
            Dictionary with token-level metrics including per-entity metrics
        """
        total_tokens = 0
        tp_tokens = 0  # Correctly identified entity tokens
        fp_tokens = 0  # Incorrectly predicted entity tokens
        fn_tokens = 0  # Missed entity tokens
        tn_tokens = 0  # Correctly identified O tokens
        
        entity_tp = 0  # Entity tokens correctly identified
        entity_fp = 0  # Non-entity tokens incorrectly identified as entity
        entity_fn = 0  # Entity tokens missed
        entity_tn = 0  # Non-entity tokens correctly identified as non-entity
        
        o_tp = 0  # O tokens correctly identified
        o_fp = 0  # Entity tokens incorrectly identified as O
        o_fn = 0  # O tokens incorrectly identified as entity
        o_tn = 0  # Entity tokens correctly identified as entity
        
        # Per-entity token-level metrics
        entity_metrics = {}  # {entity_type: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}}
        
        for doc_id in gold_annotations:
            if doc_id in system_annotations:
                gold_doc = gold_annotations[doc_id]
                sys_doc = system_annotations[doc_id]
                
                text = gold_doc.text
                if not text:
                    continue
                
                # Create BIO labels
                gold_bio = TokenLevelMetrics.create_bio_labels(text, gold_doc.get_phi())
                sys_bio = TokenLevelMetrics.create_bio_labels(text, sys_doc.get_phi())
                
                # Ensure same number of tokens
                min_len = min(len(gold_bio), len(sys_bio))
                gold_bio = gold_bio[:min_len]
                sys_bio = sys_bio[:min_len]
                
                # Count token-level metrics
                for i in range(min_len):
                    gold_label = gold_bio[i][3]  # BIO label
                    sys_label = sys_bio[i][3]    # BIO label
                    
                    total_tokens += 1
                    
                    # Convert BIO to binary (Entity vs O)
                    gold_is_entity = gold_label != "O"
                    sys_is_entity = sys_label != "O"
                    
                    # Extract entity type from BIO labels
                    gold_entity_type = gold_label.split("-")[1] if "-" in gold_label else None
                    sys_entity_type = sys_label.split("-")[1] if "-" in sys_label else None
                    
                    # Initialize entity metrics if needed
                    if gold_entity_type and gold_entity_type not in entity_metrics:
                        entity_metrics[gold_entity_type] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
                    if sys_entity_type and sys_entity_type not in entity_metrics:
                        entity_metrics[sys_entity_type] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
                    
                    if gold_is_entity and sys_is_entity:
                        # Both are entities
                        if gold_label == sys_label:
                            tp_tokens += 1
                            entity_tp += 1
                            o_tn += 1
                            # Per-entity metrics
                            if gold_entity_type:
                                entity_metrics[gold_entity_type]["tp"] += 1
                        else:
                            # Different entity types - count as entity error
                            fp_tokens += 1
                            entity_fp += 1
                            o_tn += 1
                            # Per-entity metrics
                            if gold_entity_type:
                                entity_metrics[gold_entity_type]["fn"] += 1
                            if sys_entity_type:
                                entity_metrics[sys_entity_type]["fp"] += 1
                    elif gold_is_entity and not sys_is_entity:
                        # Entity missed
                        fn_tokens += 1
                        entity_fn += 1
                        o_fp += 1
                        # Per-entity metrics
                        if gold_entity_type:
                            entity_metrics[gold_entity_type]["fn"] += 1
                    elif not gold_is_entity and sys_is_entity:
                        # Non-entity predicted as entity
                        fp_tokens += 1
                        entity_fp += 1
                        o_fn += 1
                        # Per-entity metrics
                        if sys_entity_type:
                            entity_metrics[sys_entity_type]["fp"] += 1
                    else:
                        # Both are O
                        tn_tokens += 1
                        entity_tn += 1
                        o_tp += 1
                        # Per-entity metrics: increment TN for all entity types
                        for entity_type in entity_metrics:
                            entity_metrics[entity_type]["tn"] += 1
        
        # Calculate metrics
        accuracy = (tp_tokens + tn_tokens) / total_tokens if total_tokens > 0 else 0.0
        
        # Entity class metrics (binary: Entity vs O)
        entity_precision = entity_tp / (entity_tp + entity_fp) if (entity_tp + entity_fp) > 0 else 0.0
        entity_recall = entity_tp / (entity_tp + entity_fn) if (entity_tp + entity_fn) > 0 else 0.0
        entity_f1 = (2 * entity_precision * entity_recall) / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0.0
        entity_accuracy = (entity_tp + entity_tn) / (entity_tp + entity_fp + entity_fn + entity_tn) if (entity_tp + entity_fp + entity_fn + entity_tn) > 0 else 0.0
        
        # O class metrics
        o_precision = o_tp / (o_tp + o_fp) if (o_tp + o_fp) > 0 else 0.0
        o_recall = o_tp / (o_tp + o_fn) if (o_tp + o_fn) > 0 else 0.0
        o_f1 = (2 * o_precision * o_recall) / (o_precision + o_recall) if (o_precision + o_recall) > 0 else 0.0
        o_accuracy = (o_tp + o_tn) / (o_tp + o_fp + o_fn + o_tn) if (o_tp + o_fp + o_fn + o_tn) > 0 else 0.0
        
        # Calculate per-entity token-level metrics
        per_entity_metrics = {}
        for entity_type, counts in entity_metrics.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            tn = counts["tn"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy_entity = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
            
            per_entity_metrics[entity_type] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy_entity,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn
            }
        
        return {
            "total_tokens": total_tokens,
            "accuracy": accuracy,
            "entity": {
                "precision": entity_precision,
                "recall": entity_recall,
                "f1": entity_f1,
                "accuracy": entity_accuracy,
                "tp": entity_tp,
                "fp": entity_fp,
                "fn": entity_fn,
                "tn": entity_tn
            },
            "o": {
                "precision": o_precision,
                "recall": o_recall,
                "f1": o_f1,
                "accuracy": o_accuracy,
                "tp": o_tp,
                "fp": o_fp,
                "fn": o_fn,
                "tn": o_tn
            },
            "per_entity": per_entity_metrics
        }


class BinaryEntityMetrics:
    """Class to calculate metrics for Entity (all 28 classes) vs O class."""

    @staticmethod
    def calculate_binary_metrics(tp, fp, fn, gold_annotations=None, system_annotations=None):
        """Calculate precision, recall, and F1-score for Entity vs O classification.
        
        This method calculates metrics for 2 classes:
        - Entity: All 28 entity classes grouped together
        - O: Non-entity class
        
        Args:
            tp: True positives (correctly identified entities)
            fp: False positives (incorrectly predicted entities)
            fn: False negatives (missed entities)
            gold_annotations: Dictionary of gold standard annotations (for O class calculation)
            system_annotations: Dictionary of system annotations (for O class calculation)
        """
        
        # Entity class metrics (all 28 classes grouped together)
        entity_tp = len(tp)  # Correctly identified entities
        entity_fp = len(fp)  # Incorrectly predicted entities
        entity_fn = len(fn)  # Missed entities
        
        # Count distinct entity types that appear in the dataset
        distinct_entity_types = set()
        for entity in tp:
            distinct_entity_types.add(entity[0])  # entity[0] is the entity type
        for entity in fp:
            distinct_entity_types.add(entity[0])  # entity[0] is the entity type
        for entity in fn:
            distinct_entity_types.add(entity[0])  # entity[0] is the entity type
        
        # Calculate entity class metrics
        entity_precision = entity_tp / (entity_tp + entity_fp) if (entity_tp + entity_fp) > 0 else 0.0
        entity_recall = entity_tp / (entity_tp + entity_fn) if (entity_tp + entity_fn) > 0 else 0.0
        entity_f1 = (2 * entity_precision * entity_recall) / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0.0
        
        # O class metrics
        o_tp = 0  # Tokens correctly identified as O
        o_fp = 0  # Non-O tokens incorrectly identified as O
        o_fn = 0  # O tokens incorrectly identified as non-O
        total_tokens = 0
        
        # Detailed error breakdown
        entity_to_o_errors = 0  # Entity predicted as O
        entity_to_other_entity_errors = 0  # Entity predicted as wrong entity
        o_to_entity_errors = 0  # O predicted as entity
        
        if gold_annotations and system_annotations:
            import re
            
            for doc_id in gold_annotations:
                if doc_id in system_annotations:
                    gold_doc = gold_annotations[doc_id]
                    sys_doc = system_annotations[doc_id]
                    
                    # Get text and tokenize
                    text = gold_doc.text
                    if not text:
                        continue
                        
                    tokens = []
                    for match in re.finditer(r'\S+', text):
                        tokens.append((match.group(), match.start(), match.end()))
                    
                    # Get entity spans
                    gold_spans = gold_doc.get_phi()
                    sys_spans = sys_doc.get_phi()
                    
                    # Create BIO labels for gold and system
                    gold_labels = ["O"] * len(tokens)
                    sys_labels = ["O"] * len(tokens)
                    
                    # Mark entities in gold
                    for label, start, end in gold_spans:
                        for i, (token, token_start, token_end) in enumerate(tokens):
                            if token_end <= start:
                                continue
                            if token_start >= end:
                                break
                            if token_start >= start and token_end <= end:
                                if gold_labels[i] == "O":
                                    gold_labels[i] = "B-" + label
                                elif gold_labels[i].startswith("B-") or gold_labels[i].startswith("I-"):
                                    gold_labels[i] = "I-" + label
                    
                    # Mark entities in system
                    for label, start, end in sys_spans:
                        for i, (token, token_start, token_end) in enumerate(tokens):
                            if token_end <= start:
                                continue
                            if token_start >= end:
                                break
                            if token_start >= start and token_end <= end:
                                if sys_labels[i] == "O":
                                    sys_labels[i] = "B-" + label
                                elif sys_labels[i].startswith("B-") or sys_labels[i].startswith("I-"):
                                    sys_labels[i] = "I-" + label
                    
                    # Count O class metrics and detailed errors
                    for i in range(len(tokens)):
                        if gold_labels[i] == "O" and sys_labels[i] == "O":
                            o_tp += 1
                        elif gold_labels[i] == "O" and sys_labels[i] != "O":
                            o_fn += 1
                            o_to_entity_errors += 1  # O predicted as entity
                        elif gold_labels[i] != "O" and sys_labels[i] == "O":
                            o_fp += 1
                            entity_to_o_errors += 1  # Entity predicted as O
                        elif gold_labels[i] != "O" and sys_labels[i] != "O":
                            # Both are entities, check if they match
                            if gold_labels[i] != sys_labels[i]:
                                entity_to_other_entity_errors += 1  # Entity predicted as wrong entity
                    
                    total_tokens += len(tokens)
        
        # Calculate O class metrics
        o_precision = o_tp / (o_tp + o_fp) if (o_tp + o_fp) > 0 else 0.0
        o_recall = o_tp / (o_tp + o_fn) if (o_tp + o_fn) > 0 else 0.0
        o_f1 = (2 * o_precision * o_recall) / (o_precision + o_recall) if (o_precision + o_recall) > 0 else 0.0
        
        # Calculate error breakdown percentages
        total_entity_errors = entity_to_o_errors + entity_to_other_entity_errors
        total_o_errors = o_to_entity_errors
        
        # For cases without full annotations, estimate from fp and fn
        if total_entity_errors == 0 and (len(fp) > 0 or len(fn) > 0):
            # Estimate: assume fp are mostly entity_to_o and fn are mostly entity_to_other
            entity_to_o_errors = len(fn)  # False negatives are typically missed entities
            entity_to_other_entity_errors = len(fp)  # False positives are typically wrong entity types
            total_entity_errors = entity_to_o_errors + entity_to_other_entity_errors
        
        entity_to_o_percentage = entity_to_o_errors / total_entity_errors if total_entity_errors > 0 else 0.0
        entity_to_other_entity_percentage = entity_to_other_entity_errors / total_entity_errors if total_entity_errors > 0 else 0.0
        
        return {
            "entity": {
                "precision": entity_precision,
                "recall": entity_recall,
                "f1": entity_f1,
                "tp": entity_tp,
                "fp": entity_fp,
                "fn": entity_fn
            },
            "o": {
                "precision": o_precision,
                "recall": o_recall,
                "f1": o_f1,
                "tp": o_tp,
                "fp": o_fp,
                "fn": o_fn
            },
            "error_breakdown": {
                "entity_to_o_errors": entity_to_o_errors,
                "entity_to_other_entity_errors": entity_to_other_entity_errors,
                "o_to_entity_errors": o_to_entity_errors,
                "entity_to_o_percentage": entity_to_o_percentage,
                "entity_to_other_entity_percentage": entity_to_other_entity_percentage,
                "total_entity_errors": total_entity_errors,
                "total_o_errors": total_o_errors
            },
            "distinct_entity_types": len(distinct_entity_types),
            "total_tokens": total_tokens,
            "has_o_metrics": gold_annotations is not None and system_annotations is not None,
            "has_error_breakdown": total_entity_errors > 0 or total_o_errors > 0
        }

