from typing import no_type_check

# search object
search_list = list(range(1,10001))
# search value
obj = 60

def binary_search(search_list, obj):
    left = 0 # most left position
    right = len(search_list) - 1 # most right position
    count = 0
    while left <= right:
        mid = (left + right) // 2 # middle index
        print(mid)
        mid_value = search_list[mid] # middle value
        if obj < mid_value:
            # search value is more left 
            right = mid - 1
            count += 1
            continue
        elif obj > mid_value:
            # search value is more right
            left = mid + 1
            count += 1
            continue

        return True, count
    
    return False

_, counts = binary_search(search_list, obj)
print(counts) # show search counts
