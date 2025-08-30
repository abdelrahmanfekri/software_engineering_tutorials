package practice;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

public class Main {
    public static void main(String[] args) {
        // test minimumWindowReachSum
        int[] nums = {2, 3, 1, 2, 4, 3};
        int targetSum = 7;
        int result = minimumWindowReachSum(nums, targetSum);
        System.out.println(result);
        // should print 2
    }

    public static List<Integer> topKFreq(int[] nums, int k){
        Map<Integer, Integer> frequencyByNumber = new HashMap<>();
        for(int value : nums){
            frequencyByNumber.put(value, frequencyByNumber.getOrDefault(value, 0) + 1);
        }

        PriorityQueue<int[]> minHeap = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
        for(Map.Entry<Integer, Integer> entry: frequencyByNumber.entrySet()){
            minHeap.offer(new int[]{entry.getKey(), entry.getValue()});
            if(minHeap.size() > k) minHeap.poll();
        }

        List<Integer> result = new ArrayList<>();
        while(!minHeap.isEmpty()){
            result.add(minHeap.poll()[0]);
        }
        Collections.reverse(result);
        return result;
    }

    public static int countUniquePairs(int[] nums, int targetSum){
        Set<Integer> seenValues = new HashSet<>();
        Set<String> uniquePairs = new HashSet<>();
        for(int currentValue : nums){
            int needed = targetSum - currentValue;
            if(seenValues.contains(needed)){
                uniquePairs.add(Math.min(currentValue, needed) + "," + Math.max(currentValue, needed));
            }
            seenValues.add(currentValue);
        }
        return uniquePairs.size();
    }

    public static int longestSubstringWithAtMostKDistinct(String s, int k){
        Map<Character, Integer> charFrequency = new HashMap<>();
        int left = 0;
        int right = 0;
        int maxLength = 0;
        while(right < s.length()){
            charFrequency.put(s.charAt(right), charFrequency.getOrDefault(s.charAt(right), 0) + 1);
            while(charFrequency.size() > k){
                charFrequency.put(s.charAt(left), charFrequency.get(s.charAt(left)) - 1);
                if(charFrequency.get(s.charAt(left)) == 0) charFrequency.remove(s.charAt(left));
                left++;
            }
            maxLength = Math.max(maxLength, right - left + 1);
            right++;
        }
        return maxLength;
    }

    public static int minimumWindowReachSum(int[] nums, int targetSum){
        int left = 0;
        int right = 0;
        int minLength = Integer.MAX_VALUE;
        int currentSum = 0;
        while(right < nums.length){
            currentSum += nums[right++];
            while(currentSum >= targetSum){
                minLength = Math.min(minLength, right - left + 1);
                currentSum -= nums[left++];
            }
        }
        return minLength == Integer.MAX_VALUE ? 0 : minLength;
    }

}
