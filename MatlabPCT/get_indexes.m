function [ start_index, end_index ] = get_indexes( tid, problem_size, num_threads )
    tid = tid - 1;
    block_size = floor(problem_size / num_threads);
    left_over = mod(problem_size, num_threads);
    start_index = block_size * tid;
    
    if tid >= left_over
        start_index = start_index + left_over;
    else
        start_index = start_index + tid;
    end
    
    if tid < left_over
        end_index = start_index + block_size + 1;
    else
        end_index = start_index + block_size;
    end
    
    start_index = start_index + 1;
end