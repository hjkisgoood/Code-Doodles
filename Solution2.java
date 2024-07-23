package bupt;


import java.util.Arrays;

import static java.util.Arrays.sort;

public class Solution2 {

    //贪心算法
    public int findContentChildren(int[] g, int[] s) {
        sort(g);
        sort(s);
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



    public boolean canJump(int[] nums) {
        int len = nums.length;
        if(len == 1) return true;
        int cover = 0;
        for(int i = 0; i < len; i++){
            int tem = nums[i] + i;
            cover = Math.max(cover, tem);
            if(cover >= len - 1){
                return true;
            }
            if(cover <= i){
                return false;
            }
        }
        return true;
    }//55. 跳跃游戏

    public int jump(int[] nums) {
        if( nums.length ==0 || nums.length == 1)   return 0;
        int count = 0;
        int curDistance = 0;
        int maxDistance = 0;
        for(int i = 0; i < nums.length; i++){
            maxDistance = Math.max(maxDistance, nums[i] + i);
            if(maxDistance >= nums.length - 1){
                count++;
                break;
            }
            //走到当前最大区域更新下一步
            if(i == curDistance){
                curDistance = maxDistance;
                count++;
            }
        }
        return count;
    }//45.跳跃游戏2

    public int largestSumAfterKNegations(int[] nums, int k) {
        sort(nums);
        int sum = 0;
        int index = 0;
        for(int i = 0; i < nums.length; i++){
            if(nums[i] < 0 && k > 0){
                nums[i] = -nums[i];
                index = i;
                k--;
            }
        }
        if(k == 0 || k % 2 == 0) {
            return Arrays.stream(nums).sum();
        }else if(index != nums.length - 1){
            if(nums[index + 1] > nums[index]){
                nums[index] = -nums[index];
            }else {
                nums[index + 1] = -nums[index + 1];
            }
            return Arrays.stream(nums).sum();
        }else {
            nums[index] = -nums[index];
            return Arrays.stream(nums).sum();
        }
    }//1005.k次取反后最大化的数组和


    }
