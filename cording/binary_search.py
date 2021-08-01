from typing import no_type_check

# search object
search_list = list(range(1,10001))
# search value
obj = 60

def binary_search(search_list, obj):
    low = 0 #
    high = len(search_list) - 1
    count = 0
    while low <= high:
        mid = (low + high) // 2 # middle index
        print(mid)
        mid_value = search_list[mid] # middle value
        if obj < mid_value:
            # search value is more left 
            high = mid - 1
            count += 1
            continue
        elif obj > mid_value:
            # search value is more right
            low = mid + 1
            count += 1
            continue

        return True, count
    
    return False

_, counts = binary_search(search_list, obj)
print(counts) # show search counts
