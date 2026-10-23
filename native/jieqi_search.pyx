# cython: language_level=3, infer_types=False, annotation_typing=False, boundscheck=True, wraparound=True
"""Node methods only; Python objects, sampling helpers and operation order retained."""

API_VERSION = 1
SEARCH_VERSION = 3
SOURCE_SHA256 = NATIVE_SOURCE_SHA256
METHODS_SHA256 = NATIVE_METHODS_SHA256


def initialize(module):
    if module.SEARCH_VERSION != SEARCH_VERSION or module.MOVE_SIZE != 2550:
        raise ImportError("incompatible native search version")


def initialize_node(self, prior, player_to_move, move=None, is_chance=False):
    self.prior = prior
    self.player_to_move = player_to_move
    self.visits = 0
    self.value_sum = 0.0
    self.children = {}
    self.public_outcomes = {}
    self.decision_nodes = {}
    self.move = move
    self.is_chance = is_chance
    self.reveal_moved = 0
    self.reveal_captured = 0
    self.chance_counts = None


def expand(self, legal_moves, priors, chance_specs=None, chance_k=0, rng=None):
    # Each Python class retains its own module context, including CLI/spawn imports.
    cdef object module = self._native_search_module
    cdef object total = 0.0
    cdef object one = 1.0
    if not legal_moves:
        return
    for move in legal_moves:
        total += float(priors[module.move_to_index(move)])
    if total <= 0.0:
        move_priors = {move: one / len(legal_moves) for move in legal_moves}
    else:
        move_priors = {move: float(priors[module.move_to_index(move)]) / total for move in legal_moves}
    for move in legal_moves:
        spec = chance_specs.get(move) if chance_specs else None
        moved_outcomes = (
            (module.sample_reveal_outcomes(spec["moved_counts"], chance_k, rng)
             if spec["moved_counts"] is not None else [(0, 1.0)])
            if spec is not None and rng is not None
            else []
        )
        if moved_outcomes:
            child = module.PUCTNode(move_priors[move], 1 - self.player_to_move, move=move, is_chance=True)
            child.chance_counts = tuple(
                tuple(counts) if counts is not None else None
                for counts in (spec["moved_counts"], spec["capture_counts"])
            )
            capture_outcomes = (
                module.sample_reveal_outcomes(spec["capture_counts"], chance_k, rng)
                if spec["capture_counts"] is not None
                else []
            )
            for index, (reveal_moved, moved_prob) in enumerate(moved_outcomes):
                if capture_outcomes:
                    layer2 = module.PUCTNode(moved_prob, 1 - self.player_to_move, is_chance=True)
                    layer2.reveal_moved = reveal_moved
                    for sub_index, (reveal_captured, captured_prob) in enumerate(capture_outcomes):
                        outcome_node = module.PUCTNode(captured_prob, 1 - self.player_to_move)
                        outcome_node.reveal_moved = reveal_moved
                        outcome_node.reveal_captured = reveal_captured
                        layer2.children[sub_index] = outcome_node
                    child.children[index] = layer2
                else:
                    outcome_node = module.PUCTNode(moved_prob, 1 - self.player_to_move)
                    outcome_node.reveal_moved = reveal_moved
                    child.children[index] = outcome_node
            self.children[move] = child
        else:
            self.children[move] = module.PUCTNode(move_priors[move], 1 - self.player_to_move, move=move)
