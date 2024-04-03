package bupt;

public class Solution {
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


    }
