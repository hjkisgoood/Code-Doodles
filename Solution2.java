package org.example;


import java.net.StandardSocketOptions;
import java.util.*;
import Code-Doodles/TreeNode.java;
import static java.util.Arrays.binarySearch;
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

    public int canCompleteCircuit(int[] gas, int[] cost) {
        int curSum = 0;
        int totalSum  = 0;
        int index = 0;
        for(int i = 0; i<gas.length;i++){
            curSum += gas[i] - cost[i];
            totalSum += gas[i] - cost[i];
            if(curSum < 0){
                index = (i + 1) % gas.length;
                curSum = 0;
            }
        }
        if(totalSum < 0) return -1;
        return index;

    }//134.加油站

    public int candy(int[] ratings) {
        int [] candy = new int[ratings.length];
        candy[0] = 1;
        for(int i = 1;i < ratings.length; i++){//右边比左边分高的孩子多拿一颗
            if(ratings[i] > ratings[i-1]){
                candy[i] = candy[i-1] + 1;
            }else {
                candy[i] = 1;
            }
        }
        for(int i = ratings.length - 2;i >= 0; i--){//左边比右边多分的孩子多拿一颗
            if(ratings[i] > ratings[i + 1]){
                candy[i] = candy[i + 1] + 1;
            }
        }
        int sum = 0;
        for(int i : candy){
            sum += i;
        }
        return sum;

    }//135.分发糖果

    public boolean lemonadeChange(int[] bills) {
        int five = 0;
        int ten = 0;

        for(int bill : bills){
            if(bill == 5){
                five++;
            }else if(bill == 10){
                five--;
                ten++;
            }else {
                if(ten > 0){
                    ten--;
                    five--;
                }else {
                    five = five - 3;
                }
            }
            if(five < 0 || ten < 0){
                return false;
            }

        }
        return true;
    }//860.柠檬水找零

    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, (a, b) -> {
            if(a[0] == b[0]) {
                return a[1] - b[1];
            }else {
                return b[0] - a[0];
            }
        });

        LinkedList<int[] > que = new LinkedList<>();
        for(int []p : people){
            que.add(p[1], p);
        }
        return que.toArray(new int[people.length][]);

    }//406.根据身高重建队列

    public int findMinArrowShots(int[][] points) {
        //按照气球起点排序
        //起点一样按照终点排序
        Arrays.sort(points, (a, b) -> Integer.compare(a[0], b[0]));
        int count = 1;
        System.out.println(Arrays.deepToString(points));
        for(int i =1; i < points.length; i++){
            if(points[i][0] > points[i - 1][1]){//球不挨着
                count++;//没有挨着就加一根箭
            }else{//挨着就射爆，并且将挨着的气球的终点更新为上一个球的终点
                points[i][1] = Math.min(points[i][1], points[i - 1][1]); // 更新重叠气球最小右边界
            }
        }
        return count;


    }//452.用最少的箭射爆气球

    public int eraseOverlapIntervals(int[][] intervals) {
        Arrays.sort(intervals, (a, b) ->{
            if(a[1] != b[1]){
                return Integer.compare(a[1], b[1]);
            }else {
                return Integer.compare(b[0], a[0]);
            }
        });
        System.out.println(Arrays.deepToString(intervals));

        int count = 1;
        for(int i = 1; i < intervals.length; i++){
            if(intervals[i][0] < intervals[i - 1][1]){
                intervals[i][1] = Math.min(intervals[i - 1][1], intervals[i][1]);
            }else {
                count++;
            }
        }
        return intervals.length - count;

    }//435.无重叠区间

    public List<Integer> partitionLabels(String s) {
        List<Integer> res = new LinkedList<>();
        int []edge = new int[26];
        char[] chars = s.toCharArray();
        for(int i = 0; i< chars.length; i++){
            edge[chars[i] - 'a'] = i;
        }
        int idx = 0;
        int last = -1;
        for (int i = 0;i< chars.length; i++){
            idx = Math.max(idx, edge[chars[i] - 'a']);
            if(i == idx){
                res.add(i - last);
                last = i;
            }
        }
        return res;


    }//763.划分字母边界

    public int[][] merge(int[][] intervals) {
        List<int[]> res = new ArrayList<>();
        Arrays.sort(intervals, Comparator.comparingInt(x -> x[0]));
        int start = intervals[0][0];
        int rightBound = intervals[0][1];
        for(int i = 1; i < intervals.length; i++){
            //左边界大于右边界
            //找到不重叠区间，更新左右边界
            if(intervals[i][0] > rightBound){
                res.add(new int[]{start, rightBound});
                start = intervals[i][0];
                rightBound = intervals[i][1];
            }else {
                //重叠区间，更新右边界
                rightBound = Math.max(rightBound, intervals[i][1]);
            }
        }
        //最后一个区间没有处理，这里添加上
        res.add(new int[]{start, rightBound});
        System.out.println(Arrays.deepToString(res.toArray(new int[res.size()][])));
        return res.toArray(new int[res.size()][]);


    }//56.合并区间

    public int monotoneIncreasingDigits(int n) {
        String s = String.valueOf(n);
        char[] chars = s.toCharArray();
        int start = s.length();
        for(int i = s.length() - 2;i >= 0; i--){
            if(chars[i] > chars[i + 1]){
                chars[i]--;
                start = i + 1;
            }
        }
        for(int i = start;i < s.length(); i++){
            chars[i] = '9';
        }
        int res = 0;
        for (char aChar : chars) {
            res = res * 10 + aChar - '0';
        }
        return res;
    }//738.单调递增的数字

    public int minCameraCover(TreeNode root) {
        int []res = new int[1];
        if(minCameraCoverHelp(root, res) == 0){
            res[0]++;
        }
        return res[0];

    }//968.监控二叉树
    private int minCameraCoverHelp(TreeNode root, int[] res) {
        if(root == null){return 2;}//空节点默认有覆盖了
        int left = minCameraCoverHelp(root.left);
        int right = minCameraCoverHelp(root.right);
    }

//动态规划
    public int fib(int n) {
        if(n == 0){
            return 0;
        }
        int [] dp = new int[n+1];
        dp[0] = 0;
        dp[1] = 1;
        for(int i = 2;i < n+1; i++){
            dp[i] = dp[i-2] + dp[i-1];
        }
        return dp[n];
    }//509.斐波那契

    public int climbStairs(int n) {
        if(n == 1){
            return 1;
        }
        int [] dp = new int[n];
        dp[0] = 1;
        dp[1] = 2;
        for(int i = 2;i < n; i++){
            dp[i] = dp[i-2] + dp[i-1];
        }
        return dp[n-1];
    }//70.爬楼梯

    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        int [] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 0;
        for(int i = 2;i < n+1; i++){
            dp[i] = Math.min(dp[i-2] + cost[i-2], dp[i-1] + cost[i-1]);
        }
        return dp[n];

    }//746.使用最小话费爬楼梯

    public int uniquePaths(int m, int n) {
        int [][]dp = new int[m][n];
        for(int  i = 0; i < m; i++){
            dp[i][0] = 1;
        }
        for(int i = 0;i < n; i++){
            dp[0][i] = 1;
        }
        for(int i = 1;i < m; i++){
            for(int j = 1; j < n; j++){
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m-1][n-1];


    }//62.不同路径
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int [][] dp = new int[m][n];
        if(obstacleGrid[0][0] == 1 ){
            System.out.println("test");
            return 0;
        }
        if(obstacleGrid[m-1][n-1] == 1){
            System.out.println("test");
            return 0;
        }
        System.out.println(obstacleGrid[m-1][n-1]);


        for(int i = 0;i < m && obstacleGrid[i][0] == 0;i++){
            dp[i][0] = 1;
        }
        for(int i = 0;i < n && obstacleGrid[0][i] == 0;i++){
            dp[0][i] = 1;
        }
        for(int i = 1 ; i < m; i++){
            for(int j = 1; j < n; j++){
                if(obstacleGrid[i][j] == 1){
                    dp[i][j] = 0;
                }else {
                    dp[i][j] = dp[i-1][j] + dp[i][j-1];
                }
            }
        }
        return dp[m-1][n-1];


    }//63.不同路径

    public int integerBreak(int n) {


        //return integerBreakHelp1(n);
        return integerBreakHelp2(n);


    }//343.整数拆分
    private int integerBreakHelp1(int n){
        if(n == 2) return 1;
        int n3 = n / 3;
        int n2 = 0;
        int left3 = n % 3;
        if(left3 == 1){
            n2 = 2;
            n3--;
        }else if(left3 == 2){
            n2 = 1;
        }
        return (int) (Math.pow(3,n3) * Math.pow(2,n2));
    }
    private int integerBreakHelp2(int n){
        int [] dp = new int[n + 1];
        dp[2] = 1;
        for(int i = 3; i <= n; i++ ){
            for(int j = 1; 2 * j <= i ;j++){
                dp[i] = Math.max(dp[i], Math.max(j * (i - j), j * dp[i - j]));
            }
        }
        return dp[n];


    }
    public int numTrees(int n) {
        int []dp = new int[ n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for(int i = 2; i<= n; i++){
            for(int j = 0; j < i; j++){
                //选取i作为根节点，该树的个数就等于左子树的个数乘以右子树的个数
                //把1-n的所有根节点遍历一遍
                dp[i] += dp[j] * dp[i-j-1];
            }
        }
        return dp[n];
    }//96.不同的搜索树





























    }













