def high_minimum_gain(s, x, y):
    if (x > y):
        modified_s, score1 = remove_ab(s, x)
        _, score2 = remove_ba(modified_s,y)
        return score1 + score2
    else:
        modified_s, score1 = remove_ba(s, y)
        _, score2 = remove_ab(modified_s,x)
        return score1 + score2
    
    

def remove_ab(s, x):
    score = 0
    
    i = 0

    while i < len(s):
        if (s[i] == 'a' and s[i+1] == 'b'):
            low = i
            high = i+1

            while (low > 0 and s[low-1] == 'a' and s[i+1] == 'b'): 
                low -= 1
                high += 1

             
            s = s[:low] + s[high+1:]
            score += ((high+1 - low) // 2 ) * x; 
            i = low+1
        else:
            i+=1     
         
    
    return s, score




def remove_ba(s, y):
    score = 0
    
    i = 0

    while i < len(s):
        if (s[i] == 'b' and s[i+1] == 'a'):
            low = i
            high = i+1

            while (low > 0 and s[low-1] == 'b' and s[i+1] == 'a'): 
                low -= 1
                high += 1

            print(s, high, low)
            s = s[:low] + s[high+1:]
            score += ((high+1 - low) // 2) * y; 
            i = low+1
        else:
            i+=1     
         
    
    return s, score


print(high_minimum_gain("cdbcbbaaabab", 4, 5))

