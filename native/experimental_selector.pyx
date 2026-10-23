# cython: language_level=3, infer_types=False, annotation_typing=False, boundscheck=True, wraparound=True
"""Object-arithmetic PUCT selector; never imported by production."""

API_VERSION = 1
SOURCE_SHA256 = NATIVE_SOURCE_SHA256
CORE_SHA256 = NATIVE_CORE_SHA256
CODE_SHA256 = NATIVE_CODE_SHA256
MODULE = ORIGINAL = NODE_CLASS = None


def initialize(module, fingerprint):
    global MODULE, ORIGINAL, NODE_CLASS
    if fingerprint != CODE_SHA256:
        raise ImportError("experimental selector source/function fingerprint mismatch")
    if MODULE is not None:
        if MODULE is not module:
            raise ImportError("experimental selector cannot serve another module")
        return
    MODULE = module
    NODE_CLASS = module.PUCTNode
    ORIGINAL = module.PUCTNode.select_child


def supports(node):
    if MODULE is None:
        raise RuntimeError("initialize experimental selector first")
    return (type(node) is NODE_CLASS and type(node.children) is dict
            and MODULE.PUCTNode is NODE_CLASS and "sum" not in MODULE.__dict__)


def select_child(node, c_puct):
    cdef object total_visits, best_score, best_move, best_child, sqrt_total
    cdef object move, child, u, score
    if not supports(node):
        return ORIGINAL(node, c_puct)
    # Keep the original generator sum, object arithmetic and strict tie comparison.
    total_visits = sum(child.visits for child in node.children.values())
    best_score = -1e9
    best_move = None
    best_child = None
    sqrt_total = MODULE.math.sqrt(total_visits + 1.0)
    for move, child in node.children.items():
        u = c_puct * child.prior * sqrt_total / (1 + child.visits)
        score = -child.q_value() + u
        if score > best_score:
            best_score = score
            best_move = move
            best_child = child
    return best_move, best_child
