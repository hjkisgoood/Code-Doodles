package bupt;

import java.util.Arrays;

public class Solution2 {

    //贪心算法
    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int start = 0;
        int count = 0;
        for(int i = 0; i < s.length && start < g.length; i++){
            if(s[i] >= g[start]){
                count++;
                start++;
            }
        }
        return count;
    }//455.分发饼干

    public int wiggleMaxLength(int[] nums) {
        if(nums.length <= 1) return nums.length;
        int curDiff = 0;
        int preDiff = 0;
        int count = 1;
        for(int i = 1; i < nums.length; i++){
            curDiff = nums[i] - nums[i-1];
            if((curDiff > 0 && preDiff <= 0) || (curDiff < 0 && preDiff >= 0)){
                preDiff = curDiff;
                count++;
            }
        }
        return count;
    }//376.摆动序列

    public int maxSubArray(int[] nums) {
        int maxSum = nums[0];
        int temSum = nums[0];
        for(int i = 1; i < nums.length; i++){
            if(temSum < 0){
                temSum = nums[i];
            }else {
                temSum += nums[i];
            }
            if(temSum > maxSum){
                maxSum = temSum;
            }

        }
        return maxSum;
    }//53.最大子数组和

    public int maxProfit(int[] prices) {
        int maxProfit = 0;
        int [] dis = new int[prices.length - 1];
        for(int i = 1; i < prices.length; i++){
            dis[i-1] = prices[i] - prices[i-1];
        }
        for(int profit : dis){
            if(profit > 0){
                maxProfit += profit;
            }
        }
        return maxProfit;
    }//122. 买卖股票的最佳时机 II


    }
