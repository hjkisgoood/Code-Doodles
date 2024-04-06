package bupt;

import java.util.HashMap;
import java.util.List;

public class Solution {
    //二分查找
    public int searchInsert(int[] nums, int target) {
        int n = nums.length;
        int left = 0,right = n-1;
        while(left <= right){//左闭右闭
            int mid = left +(right - left)/2;//防溢出
            if (nums[mid] > target){
                right = mid-1;
            }else if (nums[mid] < target){
                left = mid + 1 ;
            }else return mid;//if(nums[mid] == target)

        }
        return right + 1;
    }//二分查找,力扣35题插入排序

    public int[] searchRange(int[] nums, int target) {
        int left = getLeftBorder(nums,target);
        int right = getRightBorder(nums,target);
        int[] result = new int [2];
        if(right == -2 || left == -2) {
            result[0] = -1;result[1] = -1;
            return result;
        }
        if(right  - left > 1) {
            result[0] = left + 1;
            result[1] = right -1;
            return result;
        }
        return new int[]{-1, -1};


    }//力扣34题在排序数组中查找元素的第一个和最后一个位置

    private int getRightBorder(int nums[], int target){
        int left = 0;
        int right = nums.length - 1;
        int  rightBorder = -2;
        while(left <= right){
            int mid = left + (right -left) / 2;
            if(nums[mid] > target){
                right = mid -1;
            }else {
                left = mid + 1;
                rightBorder = left;
            }
        }
        return rightBorder;
    }
    private int getLeftBorder(int nums[], int target){
        int left = 0;
        int right = nums.length -1;
        int leftBorder = -2;
        while(left <= right){
            int mid = left + (right - left) / 2;
            if((nums[mid] >= target)){
                right = mid -1;
                leftBorder = right;
            }else {
                left = mid + 1;
            }
        }
        return  leftBorder;
    }

    public int mySqrt(int x) {
        long left = 1;
        long right = x;
        while(left <= right){
            long mid = left + (right - left)/2;
            if(mid * mid > x) {
                right = mid - 1;
            }else left = mid + 1;
        }
        return (int)right;

    }//69力扣,x的平方根

    public boolean isPerfectSquare(int num) {
        long right = num;
        long left = 0;
        while(left <= right){
            long mid = (left + right)/2;
            if(mid * mid == num) return true;
            else if (mid * mid < num) {
                left = mid + 1;
            }else right = mid - 1;

        }
        return false;
    }//367有效的完全平方数
//双指针
    public int removeDuplicates(int[] nums) {
        int j = 1;
        for(int i =1; i< nums.length;i++){
            if(nums[i] != nums[i-1]){
                nums[j] = nums[i];
                j++;
            }
        }
        return j;
    }//26删除重复元素
    public void moveZeroes(int[] nums) {
        for(int i = 0,j=0;i < nums.length;i++){
            if(nums[i] == 0) continue;
            else {
                int temp = nums[j];
                nums[j] = nums[i];
                nums[i] = temp;
                j++;
            }
        }

    }//283移动0

    public boolean backspaceCompare(String s, String t) {
        StringBuilder sBack = new StringBuilder();
        StringBuilder tBack = new StringBuilder();
        for(int i = 0,j=0;i < s.length();i++){
            char c = s.charAt(i);
            if(c != '#') {
                sBack.append(c);
                j++;
            } else {
                if(j > 0){
                    sBack.delete(j-1,j);
                    j--;
                }
            }
        }
        for(int i = 0,j=0;i < t.length();i++){
            char c = t.charAt(i);
            if(c != '#') {
                tBack.append(c);
                j++;
            } else {
                if(j > 0){
                    tBack.delete(j-1,j);
                    j--;
                }
            }
        }
        return sBack.toString().equals(tBack.toString());


    }//844比较含退格的字符串

    /*public int[] sortedSquares(int[] nums) {
        int mid = 0;
        int minSqu = Integer.MAX_VALUE;
        for(int i = 0;i<nums.length;i++){
            nums[i] = nums[i] * nums[i];
            if(nums[i] < minSqu){
                mid = i;
                minSqu = nums[i];
            }
        }
        int [] newNums = new int[nums.length];
        int left = mid-1,right = mid+1;
        newNums[0] =nums[mid];
        int index = 1;
        while(right < nums.length || left >= 0){
            if(left >= 0 &&right < nums.length && nums[left] <= nums[right]){
                newNums[index] = nums[left];
                left--;
                index++;
            } else if (right >=nums.length) {
                newNums[index] = nums[left];
                left--;
                index++;
            } else if(right < nums.length){
                newNums[index] = nums[right];
                right++;
                index++;
            }
        }

        return newNums;
    }
    */  //977有序数组的平方

    public int[] sortedSquares(int[] nums) {
        int right = nums.length - 1;
        int left = 0;
        int[] result = new int[nums.length];
        int index = result.length - 1;
        while (left <= right) {
            if (nums[left] * nums[left] > nums[right] * nums[right]) {
                // 正数的相对位置是不变的， 需要调整的是负数平方后的相对位置
                result[index--] = nums[left] * nums[left];
                ++left;
            } else {
                result[index--] = nums[right] * nums[right];
                --right;
            }
        }
        return result;
    }//977代码随想录版本



    //滑动窗口法
    public int minSubArrayLen(int target, int[] nums) {
        int left = 0;
        int result = Integer.MAX_VALUE;
        long sum = 0;
        for(int right = 0;right < nums.length; right++){
            sum += nums[right];
            while(sum >= target){
                result = Math.min(result,right - left + 1);
                System.out.println(result);//debug部分
                sum -= nums[left];
                left++;
            }
        }
        if(result == Integer.MAX_VALUE) return 0;
        else return result;
    }//力扣209长度最小的数组


    public int totalFruit(int[] fruits) {
        int[] tempFruit =new int[2];
        tempFruit[0] = -1;
        tempFruit[1] = -1;
        int left = 0;
        int result = 0;
        int temp_sum = 0;
        for(int right = 0;right < fruits.length;right++){
            if(tempFruit[0] == -1) {//初始化数组
                tempFruit[0] = fruits[0];
                temp_sum++;
                result = 1;
                right++;
                while(right < fruits.length && fruits[right] == tempFruit[0]){
                    temp_sum++;
                    result++;
                    right++;
                }
                if(right < fruits.length && fruits[right] != tempFruit[0]){
                    temp_sum++;
                    result++;
                    tempFruit[1] = fruits[right];
                }
                continue;
            }//先装2篮子
            if(fruits[right] == tempFruit[0] || fruits[right] == tempFruit[1]){

                temp_sum++;
                result = Math.max(result, temp_sum);
                continue;
            }
            else {
                tempFruit[0] = fruits[right-1];
                tempFruit[1] = fruits[right];
                temp_sum = 2;
                left = right -2;
                while (left >= 0){
                    if(fruits[left] == fruits[left+1]){
                        temp_sum++;
                        left--;
                    }else break;
                }
            }

        }


        return result;

    }//力扣904

    public String minWindow(String s, String t) {
        int sLen = s.length();
        int tLen = t.length();
        String result = "";
        HashMap<Character,Integer> match = new HashMap<>();
        HashMap<Character,Integer> window = new HashMap<>();
        if(s.length() < t.length()) return "";
        for(char c :t.toCharArray()){
            match.put(c, match.getOrDefault(c,0) + 1);
        }
        int valid = 0, start = 0, res = Integer.MAX_VALUE;
        for(int l =0,r = 0;r < s.length();r++){
            char ch = s.charAt(r);
            window.put(ch, window.getOrDefault(ch, 0) + 1);
            // 判断右边界对应的字符是否存在于 match 中，存在的话需要判断 window 中该字符的计数值是否达到要求了
            if (match.containsKey(ch)) {
                if (match.get(ch).equals(window.get(ch))) {
                    valid++;
                }
            }
            // 若 window 内的有效字符已经包括 match 了，那么就收缩窗口，更新最小长度，更新最优解
            while (valid == match.size()) {
                // 窗口[l,r]的长度更小，则更新 res 和 start
                if (r - l + 1 < res) {
                    res = r - l + 1;
                    start = l;
                }
                // 移出窗口中的左边界时，需要更新窗口中的有效字符的个数
                char charAtL = s.charAt(l);
                if (match.containsKey(charAtL)) {
                    if (match.get(charAtL).equals(window.get(charAtL))) {
                        valid--;
                    }
                }
                // l 移出窗口
                window.put(charAtL, window.get(charAtL) - 1);
                l++;
            }

        }
        return res == Integer.MAX_VALUE ? "" : s.substring(start, start + res);



    }//力扣76最小覆盖子串

    public int[][] generateMatrix(int n) {
        int loop = 0;
        int[][] res = new int[n][n];
        int start = 0;
        int count = 1;
        int i,j;
        while(loop < n/2){
            loop++;
            for(j= start;j < n -loop;j++){//left to right
                res[start][j] = count;
                count++;
            }

            for(i = start; i < n - loop; i++){ //right to down
                res[i][j] = count;
                count++;
            }

            for(;j >= loop; j--){//right to left
                res[i][j] = count;
                count++;
            }

            for(;i >= loop;i--){
                res[i][j] = count;
                count++;
            }
            start++;
        }
        if(n % 2 == 1){
            res[(n-1)/2][(n-1)/2] = count;
        }
        return  res;

    }//59螺旋矩阵

    public int[] spiralArray(int[][] array) {
        if(array.length == 0) return new int[0];
        int row = array.length,col = array[0].length;
        int[] ans = new int[row * col];
        int k = 0;
        int l =0,r = col -1,t =0,b = row -1;
        while(k < col * row){
            for(int i=l;i<=r;i++)
                ans[k++] = array[t][i];
            t++;
            for(int i=t;i<=b;i++)
                ans[k++] = array[i][r];
            r--;
            for(int i=r;i>=l && t <= b;i--)
                ans[k++] = array[b][i];
            b--;
            for(int i=b;i>=t && l <= r;i--)
                ans[k++] = array[i][l];
            l++;
        }
        return ans;
    }//LCR146螺旋遍历二维矩阵

    public ListNode removeElements(ListNode head, int val) {
        ListNode dummy =new ListNode();
        dummy.next = head;
        ListNode cur = dummy;
        while (cur.next != null){
            if(cur.next.val ==  val){
                cur.next = cur.next.next;
            }else cur = cur.next;
        }

        return dummy.next;
    }//203移除链表元素





    }
