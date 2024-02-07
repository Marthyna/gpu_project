def one_operation_mode(connection_info : dict) -> bool:
    return  not connection_info["prev"] and not connection_info["input"]

def two_branch_mode(connection_info : dict) -> bool:
    return not connection_info["prev"] and connection_info["input"]

def sequential_mode(connection_info: dict) -> bool:
    return connection_info["prev"] and not connection_info["input"]

def complete_mode(connection_info: dict ) -> bool:
    return connection_info["prev"] and connection_info["input"]