package bupt;

import com.sun.org.apache.xpath.internal.objects.XBoolean;

import java.util.*;

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

    public ListNode reverseList(ListNode head) {
        ListNode pre = null;
        ListNode cur = head;
        while(cur != null){
            ListNode temp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = temp;
        }
        return pre;
    }//206反转链表

    public ListNode swapPairs(ListNode head) {

        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode cur = dummy;
        ListNode p1 = new ListNode();
        ListNode p2 = new ListNode();
        ListNode p3 = new ListNode();
        while(cur.next != null && cur.next.next != null){

            p1 = cur.next;
            p2 = p1.next;
            p3= p2.next;
            cur.next = p2;
            p2.next = p1;
            p1.next = p3;
            cur = p1;
        }
        return dummy.next;

    }//24两两交换节点

    public ListNode removeNthFromEnd(ListNode head, int n){
        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode fast = dummy;
        ListNode slow = dummy;
        for(int i = 0;i <= n;i++){
            fast = fast.next;
        }
        while(fast != null){
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return dummy.next;

    }//19删除倒数n节点
/*  4.7号每日打卡*/

    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode curA = headA;
        ListNode curB = headB;
        int lenA = 0,lenB = 0;
        while(curA != null){
            lenA++;
            curA = curA.next;
        }
        while (curB != null){
            lenB++;
            curB = curB.next;
        }
        curA = headA;
        curB = headB;
        // 让curA为最长链表的头，lenA为其长度
        if(lenB > lenA){
            //1. swap (lenA, lenB);
            int tmplen = lenA;
            lenA = lenB;
            lenB = tmplen;
            //2. swap (curA, curB);
            ListNode tmNode = curA;
            curA = curB;
            curB = tmNode;
        }
        // 求长度差
        int gap = lenA - lenB;
        // 让curA和curB在同一起点上（末尾位置对齐）
        while (gap-- > 0) {
            curA = curA.next;
        }
        // 遍历curA 和 curB，遇到相同则直接返回
        while (curA != null) {
            if (curA == curB) {
                return curA;
            }
            curA = curA.next;
            curB = curB.next;
        }
        return null;

    }//面试题0207链表相交
    public ListNode detectCycle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while(fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
            if(slow == fast){
                ListNode index1 = fast;
                ListNode index2 = head;
                while(index2 != index1){
                    index2 = index2.next;
                    index1 = index1.next;

                }
                return index1;
            }
        }
        return null;
    }//142环形链表

    //4.10哈希表
    //hashset是不重复的哈希表
    //hashmap键值对哈希表
    public boolean isAnagram(String s, String t) {
        int[] record = new int[26];

        for(int i = 0;i < s.length();i++){
            record[s.charAt(i) - 'a']++;
        }

        for(int i =0 ;i < t.length();i++){
            record[t.charAt(i) - 'a']--;
        }
        for(int count : record){
            if(count != 0) return false;
        }
        return true;
    }//242有效的字母异味词


    public int[] intersection(int[] nums1, int[] nums2) {
        if(nums1.length == 0 || nums2.length == 0 ){
            return new int[0];
        }
        HashSet<Integer> set1 = new HashSet<>();
        HashSet<Integer> set2 = new HashSet<>();
        for(int i : nums1){
            set1.add(i);
        }
        for(int i : nums2){
            if(set1.contains(i)) set2.add(i);
        }
        int[] result = new int[set2.size()];
        int j = 0;
        for(int i : set2){
            result[j] = i;
            j++;
        }
        return result;


    }//349两个数组的交集
    public boolean isHappy(int n) {
        HashSet<Integer> record = new HashSet<>();
        while (n != 1 && !record.contains(n)){
            record.add(n);
            int n1 = 0;
            for(int j;n != 0;n = n/10){
                j = n % 10;
                n1 = n1 + j*j;
            }
            n =n1;
        }
        return n == 1;
    }//202快乐数
    public int[] twoSum(int[] nums, int target) {
        int [] res = new int[2];
        HashMap<Integer, Integer> map = new HashMap<>();
        for(int i = 0;i < nums.length;i++){
            int temp = target - nums[i];// 遍历当前元素，并在map中寻找是否有匹配的key
            if(map.containsKey(temp)){
                res[1] = i;
                res[0] = map.get(temp);
                break;
            }else map.put(nums[i], i);// 如果没找到匹配对，就把访问过的元素和下标加入到map中

        }
        return res;

    }//1两数字和

    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        int res = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        //统计两个数组中的元素之和，同时统计出现的次数，放入map
        for(int i : nums1){
            for(int j : nums2){
                int sum = i + j;
                map.put(sum, map.getOrDefault(sum,0) +1);
            }
        }

        //统计剩余的两个元素的和，在map中找是否存在相加为0的情况，同时记录次数
        for(int i : nums3){
            for(int j : nums4){
                res += map.getOrDefault(0 - i - j, 0);
            }
        }
        return res;

    }//454四数相加等于0

    public boolean canConstruct(String ransomNote, String magazine) {
        int []record = new int[26];

        for(char c : magazine.toCharArray()){//遍历magezine
            record[c - 'a']++;
        }
        for(char c : ransomNote.toCharArray()){
            record[c - 'a']--;
        }

        for(int i : record){
            if(i < 0 ) return false;
        }
        return true;

    }//383赎金信

    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        for(int i = 0;i < nums.length;i++){
            if(nums[i] > 0) return result;//a,left,right如果a>0那么left+right就也大于0
            if(i > 0 && nums[i] == nums[i-1]) continue;//去重
            int left = i+1;
            int right = nums.length-1;
            while(right > left){
                int sum = nums[i] + nums[left] + nums[right];
                if(sum > 0) right--;
                else if(sum < 0) left++;
                else {
                    result.add(Arrays.asList(nums[i] , nums[left], nums[right]));
                    while(right > left && nums[left] == nums[left+1]) left++;
                    while(right > left && nums[right] == nums[right-1]) right--;
                    left++;
                    right--;
                }
            }
        }
        return result;

    }//15三数之和


    //4.12字符串

    public void reverseString(char[] s) {
        int l = 0;
        int r = s.length-1;
        while(l < r){
            char tmp = s[l];
            s[l] = s[r];
            s[r] = tmp;
            l++;
            r--;
        }
    }//344反转链表

    public String reverseStr(String s, int k) {
        char[] ch  = s.toCharArray();
        for(int i = 0 ; i < ch.length-1;i += 2*k){
            int start = i;
            int end = Math.min(ch.length-1 , start+k-1);
            while(start < end){
                char tmp = ch[start];
                ch[start] = ch[end];
                ch[end] = tmp;
                start++;
                end--;
                System.out.println(ch[start]);
            }
        }
        return new String(ch);

    }//541反转字符串2
//字符数组转换String用new String[ch]
    public String reverseWords(String s) {
        StringBuilder sb = removeSpace(s);//删除多余空格
        System.out.println(sb);
        reverseString(sb,0,sb.length() -1);//反转
        System.out.println(sb);
        reverseEachWord(sb);
        System.out.println(sb);
        return sb.toString();

    }//151反转字符串中的单词
    private StringBuilder removeSpace(String s){
        int start = 0;
        int end = s.length() - 1;
        while(s.charAt(start) == ' ') start ++;
        while(s.charAt(end) == ' ') end--;
        StringBuilder sb = new StringBuilder();
        while(start <= end){
            char c = s.charAt(start);
            if(c != ' ' || sb.charAt(sb.length() - 1) != ' '){
                sb.append(c);
            }
            start++;
        }
        return sb;
    }
    public void reverseString(StringBuilder sb, int start, int end){
        while(start < end){
            char tmp = sb.charAt(start);
            sb.setCharAt(start, sb.charAt(end));
            sb.setCharAt(end, tmp);
            start++;
            end--;
        }
     }

    private void reverseEachWord(StringBuilder sb){
        int start = 0;
        int end = 1;
        int n = sb.length();
        while(start < n){
            while (end < n && sb.charAt(end) != ' '){
                end++;
            }
            reverseString(sb, start, end-1);
            start = end + 1;
            end = start + 1;
        }
     }
    /*
import java.util.Scanner;

class Main{
    public static void main(String[] args){
        Scanner in = new Scanner(System.in);
        String s = in.nextLine();
        StringBuilder sb = new StringBuilder();
        for(int i = 0 ;i < s.length() ;i++){
            if(s.charAt(i) >= '0' && s.charAt(i) <= '9'){
                sb.append("number");
            }else sb.append(s.charAt(i));
        }
        System.out.println(sb);
    }
}
 */ //卡码网54替换数字

    /*
     import java.util.*;

class Main{
    public static void main(String[] args){
        Scanner in = new Scanner(System.in);
        int k  = in.nextInt();
        in.nextLine();//读取换行符
        String s = in.nextLine();
        int len = s.length();
        char[] chars = new char[len];
        StringBuilder sb = new StringBuilder();
        sb.append(s.substring(len - k , len ));
        sb.append(s.substring(0, len-k));
        System.out.println(sb);
    }
}
      */ //卡码网55右旋字符串

    public int strStr(String haystack, String needle) {
        if(needle.length() == 0) return 0;

        int[] next = new int[needle.length()];
        getNext(next, needle);
        int j = -1;
        for(int i = 0;i < haystack.length(); i++){
            while (j >= 0 && haystack.charAt(i) != needle.charAt(j + 1)) j= next[j];

            if(haystack.charAt(i) == needle.charAt(j+1)) j++;

            if(j == needle.length() - 1) return (i - needle.length() + 1);
        }
        return -1;

     }//28找出字符串中第一个匹配项的下标
    private void getNext(int [] next,String s){
        int j = -1;
        next[0] = j;
        for(int i = 1; i < s.length();i++){
            while(j >= 0 && s.charAt(i) != s.charAt(j + 1)) j = next[j];

            if(s.charAt(i) == s.charAt(j + 1)) j++;

            next[i] = j;
        }
    }//创建next数组
    /*
    public int strStr(String haystack, String needle) {

        int len = needle.length();
        for(int i =0;i <= haystack.length()-len;i++){
            String str = haystack.substring(i,i+len);
            if(str.equals(needle)) return i;
        }
        return -1;
    }

     *///28找出第一个匹配项的下标 库函数版本

    public boolean repeatedSubstringPattern(String s) {
        if(s.length() == 1) return false;

        int len = s.length();
        s = " " + s;
        char[] chars = s.toCharArray();
        int[] next = new int[len+1];
        // 构造 next 数组过程，j从0开始(空格)，i从2开始
        for(int i = 2,j = 0;i <= len;i++){
            // 匹配不成功，j回到前一位置 next 数组所对应的值
            while (j > 0 && chars[i] != chars[j+1]) j = next[j];
            //匹配成功
            if( chars[i] == chars[j+1]) j++;
            // 更新 next 数组的值
            next[i] = j;
        }
        // 最后判断是否是重复的子字符串，这里 next[len] 即代表next数组末尾的值
        if(next[len] > 0 && len % (len - next[len]) == 0) return true;
        return false;


    }//459重复的子字符串

    public boolean isValid(String s) {
        Deque<Character> deque = new LinkedList<>();//这里使用栈
        char ch;
        for(int i = 0;i< s.length();i++){
            ch = s.charAt(i);
            if(ch == '(') deque.push(')');
            else if(ch == '[') deque.push(']');
            else if (ch == '{') deque.push('}');
            else if(deque.isEmpty() || deque.peek() != ch) return false;
            else deque.pop();//如果是右括号且匹配则弹出栈顶
        }
        return deque.isEmpty();//栈为空说明匹配完成
    }//20有效的括号

    public String removeDuplicates(String s) {
        ArrayDeque<Character> deque = new ArrayDeque<>();
        char ch;
        for(int i = 0;i < s.length();i++ ){
            ch = s.charAt(i);
            if(deque.isEmpty() || deque.peek() != ch) deque.push(ch);//不同则压入
            else  deque.pop();//相同弹出
        }
        String str = "";
        while (!deque.isEmpty()) str = deque.pop() + str;//利用str加法的特性保证不反转
        return str;
    }//1047删除字符串的所有相邻重复项
    /*
    public String removeDuplicates(String s) {
        // 将 res 当做栈
        // 也可以用 StringBuilder 来修改字符串，速度更快
        // StringBuilder res = new StringBuilder();
        StringBuffer res = new StringBuffer();
        // top为 res 的长度
        int top = -1;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            // 当 top > 0,即栈中有字符时，当前字符如果和栈中字符相等，弹出栈顶字符，同时 top--
            if (top >= 0 && res.charAt(top) == c) {
                res.deleteCharAt(top);
                top--;
            // 否则，将该字符 入栈，同时top++
            } else {
                res.append(c);
                top++;
            }
        }
        return res.toString();
    }
     */ //1047拿字符串直接作为栈，省去了栈还要转为字符串的操作。

    /*
      public String removeDuplicates(String s) {
        // 将 res 当做栈
        // 也可以用 StringBuilder 来修改字符串，速度更快
        // StringBuilder res = new StringBuilder();
        StringBuffer res = new StringBuffer();
        // top为 res 的长度
        int top = -1;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            // 当 top > 0,即栈中有字符时，当前字符如果和栈中字符相等，弹出栈顶字符，同时 top--
            if (top >= 0 && res.charAt(top) == c) {
                res.deleteCharAt(top);
                top--;
            // 否则，将该字符 入栈，同时top++
            } else {
                res.append(c);
                top++;
            }
        }
        return res.toString();
    }

     */ //1047拓展：双指针

    public int evalRPN(String[] tokens) {
        Deque<Integer> stack = new LinkedList<>();
        for(String s : tokens){
            if("+".equals(s)) stack.push(stack.poll() + stack.pop());
            else if("-".equals(s) ) stack.push(-stack.pop() + stack.pop());
            else if("*".equals(s)) stack.push(stack.pop() * stack.pop());
            else if("/".equals(s)) {
                int p1 = stack.pop();
                int p2 = stack.pop();
                stack.push(p2 / p1);
            }else stack.push(Integer.valueOf(s));
        }
        return stack.pop();
    }//150逆波兰表达式求值
    public int[] maxSlidingWindow(int[] nums, int k) {
        ArrayDeque<Integer> deque  = new ArrayDeque<>();
        int n = nums.length;
        int [] res = new int[n-k+1];
        int idx = 0;
        for(int i = 0;i<n;i++){
            //i是数组下标,在[i,i+k-1]找到最大值并储存
            //队列头结点需要在i,i+k-1范围内,不在则弹出
            while (!deque.isEmpty() && deque.peek() < i - k +1) deque.poll();//弹出队列头
            //保证每次放进去的数字都比末尾的大,否则弹出
            while(!deque.isEmpty() && nums[deque.peekLast()] < nums[i]) deque.pollLast();

            deque.offer(i);
            if(i >= k - 1) res[idx++] = nums[deque.peek()];

        }
        return res;
    }//239滑动窗口的最大值
    /*
    public int[] topKFrequent(int[] nums, int k){
    Map<Integer,Integer> map = new HashMap<>(); //key为数组元素值,val为对应出现次数
        for (int num : nums) {
        map.put(num, map.getOrDefault(num,0) + 1);
    }
    //在优先队列中存储二元组(num, cnt),cnt表示元素值num在数组中的出现次数
    //出现次数按从队头到队尾的顺序是从大到小排,出现次数最多的在队头(相当于大顶堆)
    PriorityQueue<int[]> pq = new PriorityQueue<>((pair1, pair2) -> pair2[1] - pair1[1]);
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {//大顶堆需要对所有元素进行排序
        pq.add(new int[]{entry.getKey(), entry.getValue()});
    }
    int[] ans = new int[k];
        for (int i = 0; i < k; i++) { //依次从队头弹出k个,就是出现频率前k高的元素
        ans[i] = pq.poll()[0];
    }
        return ans;
    }//347前k个高频元素
*/  //347前k个高频元素
    public int[] topKFrequent(int[] nums, int k) {
        // 优先级队列，为了避免复杂 api 操作，pq 存储数组
        // lambda 表达式设置优先级队列从大到小存储 o1 - o2 为从小到大，o2 - o1 反之
        PriorityQueue<int[]> pq = new PriorityQueue<>((o1, o2) -> o1[1] - o2[1]);
        int[] res = new int[k]; // 答案数组为 k 个元素
        Map<Integer, Integer> map = new HashMap<>(); // 记录元素出现次数
        for (int num : nums) map.put(num, map.getOrDefault(num, 0) + 1);
        for (Map.Entry<Integer, Integer> x : map.entrySet()) { // entrySet 获取 k-v Set 集合
            // 将 kv 转化成数组
            int[] tmp = new int[2];
            tmp[0] = x.getKey();
            tmp[1] = x.getValue();
            pq.offer(tmp);
            // 下面的代码是根据小根堆实现的，我只保留优先队列的最后的k个，只要超出了k我就将最小的弹出，剩余的k个就是答案
            if(pq.size() > k) {
                pq.poll();
            }
        }
        for (int i = 0; i < k; i++) {
            res[i] = pq.poll()[0]; // 获取优先队列里的元素
        }
        return res;
    }

    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        //preorderHelp1(root, result);//递归法
        //preorderHelp2(root, result);//迭代法
        preorderHelp3(root, result);//统一迭代法
        return result;
    }///144二叉树的前序遍历

    private void preorderHelp1(TreeNode root, List<Integer> list){
        if(root == null) return;
        list.add(root.val);
        preorderHelp1(root.left, list);
        preorderHelp1(root.right, list);
    }//递归法前序遍历
    private void preorderHelp2(TreeNode root, List<Integer> list){
        Stack<TreeNode> st = new Stack<>();
        if(root == null) return;
        st.push(root);
        while (!st.isEmpty()){
            TreeNode node = st.pop();
            list.add(node.val);
            if(node.right != null) st.push(node.right);
            if(node.left != null) st.push(node.left);
        }

    }//迭代法前序遍历
    private void preorderHelp3(TreeNode root, List<Integer> list){
        Stack<TreeNode> st = new Stack<>();
        if(root == null) return;
        st.push(root);
        while (!st.isEmpty()){
            TreeNode node = st.pop();
            if(node != null){
                if(node.right !=null) st.push(node.right);
                if(node.left != null) st.push(node.left);
                st.push(node);
                st.push(null);
            }else {
                node = st.pop();
                list.add(node.val);
            }
        }
    }//统一迭代法前序遍历

    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        //postorderHelp1(root, result);//递归法
        //postorderHelp2(root, result);//迭代法
        postorderHelp3(root, result);//统一迭代法
        return result;
    }//145后续遍历
    private void postorderHelp1(TreeNode root, List<Integer> list){
        if(root == null) return;
        postorderHelp1(root.left, list);
        postorderHelp1(root.right, list);
        list.add(root.val);
    }//递归法后序遍历

    private void postorderHelp2(TreeNode root, List<Integer> list){
        Stack<TreeNode> stack = new Stack<>();
        if(root == null) return;
        stack.push(root);
        while (!stack.isEmpty()){
            TreeNode node = stack.pop();
            list.add(node.val);
            if(node.left != null) stack.push(node.left);
            if(node.right != null) stack.push(node.right);
        }
        Collections.reverse(list);
    }//迭代法后序遍历
    private void postorderHelp3(TreeNode root, List<Integer> list){
        Stack<TreeNode> st = new Stack<>();
        if(root == null) return;
        st.push(root);
        while (!st.isEmpty()){
            TreeNode node = st.pop();
            if(node != null){
                st.push(node);
                st.push(null);//
                if(node.right !=null) st.push(node.right);
                if(node.left != null) st.push(node.left);

            }else {
                node = st.pop();
                list.add(node.val);
            }
        }
    }//统一迭代法

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        //inorderHelp1(root, result);//递归法
        //inorderHelp2(root, result);//迭代法
        inorderHelp3(root, result);//统一迭代法

        return result;

    }//94二叉树的中序遍历
    private void inorderHelp1(TreeNode root, List<Integer> list){
        if(root == null) return;
        inorderHelp1(root.left, list);
        list.add(root.val);
        inorderHelp1(root.right, list);
    }//递归法中序遍历
    private void inorderHelp2(TreeNode root, List<Integer> list){
        Stack<TreeNode> st = new Stack<>();
        if(root == null) return;
        TreeNode cur = root;
        while (cur != null || !st.isEmpty()){
            if(cur != null){
                st.push(cur);
                cur = cur.left;
            }else {
                cur = st.pop();
                list.add(cur.val);
                cur = cur.right;
            }
        }

    }//迭代法中序遍历
    private void inorderHelp3(TreeNode root, List<Integer> list){
        Stack<TreeNode> st = new Stack<>();
        if(root == null) return;
        st.push(root);
        while (!st.isEmpty()){
            TreeNode node = st.pop();
            if(node != null){
                if(node.right !=null) st.push(node.right);
                st.push(node);
                st.push(null);
                if(node.left != null) st.push(node.left);
            }else {
                node = st.pop();
                list.add(node.val);
            }
        }
    }//统一迭代法中序遍历


    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        levelOrderHelp1(root, 0, res);//递归法
        //levelOrderHelp2(root, res);//迭代法
        return res;
    }//102二叉树的层序遍历1
    private void levelOrderHelp1(TreeNode root, int deep, List<List<Integer>> res){
        if(root == null) return;
        Queue<TreeNode> que = new LinkedList<>();
        deep++;
        if(res.size() < deep){
            List<Integer> tempList = new ArrayList<>();
            res.add(tempList);
        }
        res.get(deep - 1).add(root.val);
        levelOrderHelp1(root.left, deep, res);
        levelOrderHelp1(root.right, deep, res);
    }//DFS--递归法
    private void levelOrderHelp2(TreeNode root, List<List<Integer>> res){
        if(root == null) return;
        Queue<TreeNode> que = new LinkedList<TreeNode>();
        que.offer(root);
        while(!que.isEmpty()){
            List<Integer> temList = new ArrayList<>();
            int len = que.size();
            for(int i = 0; i < len; i++){
                TreeNode tempNode = que.poll();
                temList.add(tempNode.val);
                if(tempNode.left != null) que.offer(tempNode.left);
                if(tempNode.right != null) que.offer(tempNode.right);
            }
            res.add(temList);
        }
    }//迭代法层序遍历
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> res = levelOrder(root);
        Collections.reverse(res);
        return res;
    }//107二叉树的层序遍历2
    public List<Integer> rightSideView(TreeNode root) {
        //解法一调用102层序遍历方法每次取list最后一个元素就行
        //解法2
        List<Integer> list = new ArrayList<>();
        Queue<TreeNode> que = new LinkedList<>();
        if(root == null) return list;
        que.offer(root);
        while (!que.isEmpty()){
            int lever = que.size();
            for(int i = 0;i < lever;i++){
                TreeNode poll = que.poll();
                if(poll.left != null) que.offer(poll.left);
                if(poll.right != null) que.offer(poll.right);

                if(i == lever-1)
                    list.add((poll.val));
            }
        }
        return list;
     }//199二叉树的右视图
    public List<Double> averageOfLevels(TreeNode root) {
        List<List<Integer>> list = levelOrder(root);
        List<Double> res = new ArrayList<>();
        for(List<Integer> temp: list){
            double sum = 0;
            int len = 0;
            for(int i : temp){
                sum+=i;
                len++;
            }

            double avRes = sum/(double)len;
            res.add(avRes);
        }
        return res;
    }//637二叉树的层平均值

    private class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;
        public List<Node> children;

        public Node() {}

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, List<Node> _children) {
            val = _val;
            children = _children;
        }
        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }//n叉树的定义与带next指针二叉树合二为一的
    public List<List<Integer>> levelOrder(Node root) {
        List<List<Integer>> res = new ArrayList<>();
        Queue<Node> que = new LinkedList<>();
        if(root == null) return res;
        que.offer(root);

        while (!que.isEmpty()){
            int lever = que.size();
            List<Integer> leverList = new ArrayList<>();

            for(int i = 0;i < lever;i++){
                Node tempNode = que.poll();
                leverList.add(tempNode.val);
                for(Node child : tempNode.children){
                    if(child != null) que.offer(child);
                }
            }
            res.add(leverList);
        }
        return res;
    }//迭代法n叉树层序遍历
    //429.N叉树的层序遍历
    public List<Integer> largestValues(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Queue<TreeNode> que = new LinkedList<>();
        if(root == null) return res;
        que.offer(root);
        while (!que.isEmpty()){
            int max = Integer.MIN_VALUE;//
            for(int i =que.size();i>0;i--){
                TreeNode node = que.poll();
                max = Math.max(max, node.val);
                if(node.left != null) que.offer(node.left);
                if(node.right != null) que.offer(node.right);
            }
            res.add(max);
        }
        return res;
    }//515在每个树行中找最大值


    public Node connect(Node root) {
        Queue<Node> que = new LinkedList<>();
        if(root == null) return root;
        que.offer(root);

        while(!que.isEmpty()){
            Node cur = que.poll();
            int size = que.size();
            if(cur.left != null) que.offer(cur.left);
            if(cur.right != null) que.offer(cur.right);

            for(int i = 0;i < size ;i++){
                Node next = que.poll();
                cur.next = next;
                cur = next;//当前节点指向兄弟节点并且往后移动一格
                if(cur.left != null) que.offer(cur.left);
                if(cur.right != null) que.offer(cur.right);
            }
        }
        return  root;
    }//116.填充每个节点的下一个右侧节点指针一
    /*public Node connect(Node root) {
        Queue<Node> que = new LinkedList<>();
        if(root == null) return root;
        que.offer(root);

        while(!que.isEmpty()){
            Node cur = que.poll();
            int size = que.size();
            if(cur.left != null) que.offer(cur.left);
            if(cur.right != null) que.offer(cur.right);

            for(int i = 0;i < size ;i++){
                Node next = que.poll();
                cur.next = next;
                cur = next;//当前节点指向兄弟节点并且往后移动一格
                if(cur.left != null) que.offer(cur.left);
                if(cur.right != null) que.offer(cur.right);
            }
        }
        return  root;
    }*///117.填充每个节点的下一个右侧节点指针II
    //与116代码一模一样注释一下
    public int maxDepth(TreeNode root) {
        Queue<TreeNode> que = new LinkedList<>();
        int depth = 0;
        if(root == null) return depth;
        que.offer(root);
        while (!que.isEmpty()){
            depth++;
            for(int i = que.size();i > 0; i--){
                TreeNode node = que.poll();
                if(node.left != null) que.offer(node.left);
                if(node.right != null) que.offer(node.right);
            }
        }
        return depth;
    }//102二叉树的最大深度
    public int minDepth(TreeNode root) {
        Queue<TreeNode> que = new LinkedList<>();
        int depth = 0;
        if(root == null) return depth;
        que.offer(root);
        while (!que.isEmpty()){
            depth++;
            for(int i = que.size();i > 0; i--){
                TreeNode node = que.poll();
                if(node.left == null && node.right == null) return depth;
                if(node.left != null) que.offer(node.left);
                if(node.right != null) que.offer(node.right);
            }
        }
        return depth;
    }//111.二叉树的最小深度
    public TreeNode invertTree(TreeNode root) {
        //invertTreeHelp1(root);//层序遍历解决
        invertTreeHelp2(root);//递归法
        return root;
    }//226.翻转二叉树
    private void invertTreeSwap(TreeNode root){
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
    }//交换节点左右节点
    private void invertTreeHelp1(TreeNode root){
        Queue<TreeNode> que = new LinkedList<>();
        if(root == null) return;
        que.offer(root);
        while (!que.isEmpty()){
            TreeNode node = que.poll();
            invertTreeSwap(node);
            if(node.left != null) que.offer(node.left);
            if(node.right != null) que.offer(node.right);
        }
    }//层序遍历法
    private void invertTreeHelp2(TreeNode root){
        if(root == null) return;
        invertTreeSwap(root);
        invertTreeHelp2(root.left);
        invertTreeHelp2(root.right);
    }
    public boolean isSymmetric(TreeNode root){
        if(root == null) return true;
        //return isSymmetricHelp1(root.left, root.right);
        return isSymmetricHelp2(root);
    }
    //101对称二叉树
    private boolean isSymmetricHelp1(TreeNode left, TreeNode right){
        if(left == null && right == null) return true;
        else if(left == null && right != null) return false;
        else if(left != null && right ==null) return false;
        else if(left.val != right.val) return false;
        //比较左节点的左,右节点的右,
        // 比较左节点的右,右节点的左
        // 对称比较
        return isSymmetricHelp1(left.left, right.right) && isSymmetricHelp1(left.right, right.left);
    }//递归法
    private boolean isSymmetricHelp2(TreeNode root){
        Queue<TreeNode> que = new LinkedList<>();
        que.offer(root.left);
        que.offer(root.right);
        while (!que.isEmpty()){
            TreeNode lNode = que.poll();
            TreeNode rNode = que.poll();
            if(lNode == null && rNode == null) continue;
            if(lNode == null || rNode == null || lNode.val != rNode.val) return false;

            que.offer(lNode.left);
            que.offer(rNode.right);
            que.offer(lNode.right);
            que.offer(rNode.left);
        }
        return true;
    }//迭代法
    public boolean isSameTree(TreeNode p, TreeNode q) {
        Queue<TreeNode> que = new LinkedList<>();
        que.offer(p);
        que.offer(q);
        if(q == null && p == null) return true;
        while (!que.isEmpty()){
            TreeNode left = que.poll();
            TreeNode right = que.poll();
            if(left == null && right == null) continue;
            if(left == null || right == null || left.val != right.val) return false;
            //System.out.println("test");
            que.offer(left.left);
            que.offer(right.left);
            que.offer(left.right);
            que.offer(right.right);
        }
        return true;
    }//100相同的树
    public boolean isSubtree(TreeNode root, TreeNode subRoot) {
        Queue<TreeNode> que = new LinkedList<>();
        que.offer(root);
        while (!que.isEmpty()){
            TreeNode node = que.poll();
            boolean trueOrNot = isSameTree(node, subRoot);
            if(trueOrNot) return true;
            if(node.left != null) que.offer(node.left);
            if(node.right != null) que.offer(node.right);

        }
        return false;
    }//572相同的zishu
    public int maxDepth(Node root) {
        if(root == null) return 0;
        Queue<Node> que = new LinkedList<>();
        que.offer(root);
        int depth = 0;
        while (!que.isEmpty()){
            for(int i = que.size();i>0;i--){
                Node node = que.poll();
                for(Node child : node.children) que.offer(child);
            }
            depth++;
        }
        return depth;
    }//559.n叉树的最大深度(opens new window)
    public int countNodes(TreeNode root) {
        if(root == null) return 0;
        int leftCnt = countNodes(root.left);
        int rightCnt = countNodes(root.right);
        return leftCnt + rightCnt+1;
    }//222.完全二叉树的节点个数/递归法
    public boolean isBalanced(TreeNode root) {
        //return isBalancedHelp1(root);//层序遍历调函数法,时间复杂度高
        return isBalancedHelp2(root) != -1;
    }//110.平衡二叉树
    private boolean isBalancedHelp1(TreeNode root){
        Stack<TreeNode> st = new Stack<>();
        if(root == null) return true;
        st.push(root);
        while (!st.isEmpty()){
            TreeNode node = st.pop();
            int lDepth = maxDepth(node.left);
            int rDepth = maxDepth(node.right);
            int level = lDepth - rDepth;
            if(level > 1 || level < -1) return false;
            if(node.left != null) st.push(node.left);
            if(node.right != null) st.push(node.right);
        }
        return true;
    }//层序遍历调函数法,时间复杂度高
    private int isBalancedHelp2(TreeNode root){
        if(root == null) return 0;
        int leftH = isBalancedHelp2(root.left);
        if(leftH == -1) return -1;
        int rightH = isBalancedHelp2(root.right);
        if(rightH == -1) return -1;
        //判断左右zishu高度差
        if(Math.abs(leftH - rightH) > 1) return -1;
        return Math.max(leftH, rightH) + 1;

    }

    public List<String> binaryTreePaths(TreeNode root) {
        //return binaryTreePathsHelp1(root);//回溯法1
        //return binaryTreePathsHelp2(root);//回溯法2
        return binaryTreePathsHelp3(root);//迭代法
    }//257. 二叉树的所有路径
    private List<String> binaryTreePathsHelp1(TreeNode root){
        List<String> res = new ArrayList<>();
        if(root == null) return res;
        List<Integer> paths = new ArrayList<>();
        binaryTreePathsHelp1Help(root, paths, res);
        return res;
    }//递归法1
    private void binaryTreePathsHelp1Help(TreeNode root, List<Integer> paths, List<String> res){
        paths.add(root.val);//前序遍历,中
        if(root.left == null && root.right == null){
            StringBuilder sb = new StringBuilder();
            for(int i =0;i < paths.size() - 1;i++){
                sb.append(paths.get(i)).append("->");
            }
            sb.append(paths.get(paths.size() - 1));//加入最后一个节点
            res.add(sb.toString());//加入路径
        }
        //递归与回溯
        if(root.left != null){//左
            binaryTreePathsHelp1Help(root.left, paths, res);
            paths.remove(paths.size() - 1 );//回溯
        }
        if(root.right != null){//右
            binaryTreePathsHelp1Help(root.right, paths, res);
            paths.remove(paths.size() - 1);
        }
    }

    private List<String> binaryTreePathsHelp2(TreeNode root){
        List<String> result = new ArrayList<>();
        binaryTreePathsHelp2Help(root, "", result);
        return result;
    }//回溯法二

    private void binaryTreePathsHelp2Help(TreeNode node, String s, List<String> result){
        if(node == null) return;
        if(node.left == null && node.right == null){
            result.add(new StringBuilder(s).append(node.val).toString());
            return;
        }
        String tmp = new StringBuilder(s).append(node.val).append("->").toString();
        binaryTreePathsHelp2Help(node.left, tmp, result);
        binaryTreePathsHelp2Help(node.right, tmp ,result);
    }

    private List<String> binaryTreePathsHelp3(TreeNode root){
        List<String> res = new ArrayList<>();
        if(root == null){
            return res;
        }
        Stack<Object> st = new Stack<>();
        st.push(root);
        st.push(root.val + "");
        while (!st.isEmpty()){
            String path = (String) st.pop();
            TreeNode node = (TreeNode) st.pop();
            // 遇到叶子节点
            if(node.left == null && node.right ==null){
                res.add(path);
            }
            if(node.right != null){
                st.push(node.right);
                st.push(path + "->" + node.right.val);
            }
            if(node.left != null){
                st.push(node.left);
                st.push(path + "->" + node.left.val);
            }
        }
        return res;
    }//迭代法


    public int sumOfLeftLeaves(TreeNode root) {
        //return sumOfLeftLeavesHelp1(root);//递归法
        return sumOfLeftLeavesHelp2(root);//迭代法法
    }//404左叶子之和
    private int sumOfLeftLeavesHelp1(TreeNode root){
        if(root == null) return 0;
        int left = sumOfLeftLeavesHelp1(root.left);
        int right = sumOfLeftLeaves(root.right);
        int mid = 0;
        if(root.left != null && root.left.left == null && root.left.right == null) mid = root.left.val;

        int sum = left + right + mid;
        return sum;
    }//递归法
    private int sumOfLeftLeavesHelp2(TreeNode root){
        if(root == null) return 0;
        Stack<TreeNode> st = new Stack<>();
        int result = 0;
        st.push(root);
        while (!st.isEmpty()) {
            TreeNode node = st.pop();
            if (node.left != null && node.left.left == null && node.left.right == null) result += node.left.val;

            if (node.right != null) st.push(node.right);
            if (node.left != null) st.push(node.left);
        }
        return result;
    }//迭代法

    public int findBottomLeftValue(TreeNode root) {
        Queue<TreeNode> que = new LinkedList<>();
        int result = root.val;
        que.offer(root);
        while (!que.isEmpty()){
            result = que.peek().val;
            for(int i = que.size();i > 0;i--){
                TreeNode node = que.poll();
                if(node.left != null) que.offer(node.left);
                if(node.right != null) que.offer(node.right);
            }

        }
        return result;

    }//513找树左下角的值
    //迭代法

    public boolean hasPathSum(TreeNode root, int targetSum) {
        if(root == null) return false;
        //return hasPathSumHelp1(root, targetSum - root.val);递归法
        return hasPathSumHelp2(root, targetSum);//迭代法

    }//112路径之和
    private boolean hasPathSumHelp1(TreeNode node, int count){
        if(node.left == null && node.right == null && count == 0) return true;
        if(node.left == null && node.right == null && count != 0) return false;

        if(node.left != null) {
            count -= node.left.val;
            if(hasPathSumHelp1(node.left, count)) return true;
            count += node.left.val;
        }
        if(node.right != null){
            count -= node.right.val;
            if(hasPathSumHelp1(node.right, count)) return true;
            count += node.right.val;
        }
         return false;
    }//递归法

    private boolean hasPathSumHelp2(TreeNode root, int target){
        Stack<TreeNode> st1 = new Stack<>();
        Stack<Integer> st2 = new Stack<>();
        st1.push(root);
        st2.push(root.val);

        while (!st1.isEmpty()){
            TreeNode node = st1.pop();
            int curCount = st2.pop();
            //如果是叶子
            if(node.left == null && node.right == null && curCount == target) return true;
            //不是叶子则压入
            if(node.right != null){//右
                st1.push(node.right);
                st2.push(node.right.val + curCount);
            }
            if(node.left != null){//左
                st1.push(node.left);
                st2.push(node.left.val + curCount);
            }


        }
        return false;
    }//迭代法

    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> res = new ArrayList<>();
        if(root == null) return res;

        List<Integer> path = new ArrayList<>();
        pathSumHelp(root, targetSum, res, path);
        return res;
    }//113. 路径总和ii

    private void pathSumHelp(TreeNode node, int target, List<List<Integer>> res, List<Integer> path){
        path.add(node.val);
        //遇到了叶子节点
        if(node.left == null && node.right == null){
            if(target - node.val == 0) {
                //res.add(path);//这是不行的因为java存的是引用之后回溯删除path内容,res里面也会跟着删除
                res.add(new ArrayList<>(path));
            }
           /* for(List<Integer> List : res){
                for(int i : path){
                    System.out.println(i);
                }
            }*/
            return;//叶子路径不合适则回溯,合适则返回
        }
        if(node.left != null){
            pathSumHelp(node.left, target - node.val, res, path);
            path.remove(path.size() - 1);//回溯
        }
        if(node.right != null){
            pathSumHelp(node.right, target - node.val, res, path);
            path.remove(path.size() - 1);//回溯
        }
    }//回溯法


    public TreeNode buildTree(int[] inorder, int[] postorder) {
        Map<Integer, Integer> map = new HashMap<>();
        for(int i = 0;i < inorder.length; i++){
            map.put(inorder[i], i);//map存中序数值位置
        }
        return buildTreeHelp1(inorder, 0, inorder.length, postorder, 0, postorder.length, map);

    }//106.从中序与后序遍历序列构造二叉树
    private TreeNode buildTreeHelp1(int[] inorder, int inBegin, int inEnd, int[] postorder, int poBegin, int poEnd, Map<Integer, Integer> map){
        //左闭右开区间
        if(inBegin >= inEnd || poBegin >= poEnd) return null;//不满足左闭右开
        int rootIndex = map.get(postorder[poEnd - 1]);//找到后序遍历的最后一个元素在中序遍历中的位置
        TreeNode root = new TreeNode(inorder[rootIndex]);//构造节点
        int lenOfLeft = rootIndex - inBegin;//保存中序左子树个数,用来区分后序遍历左右子树长度
        root.left = buildTreeHelp1(inorder, inBegin, rootIndex, postorder, poBegin, poBegin + lenOfLeft, map);
        root.right = buildTreeHelp1(inorder, rootIndex+1, inEnd, postorder, poBegin + lenOfLeft, poEnd-1, map);
        return root;
    }//中序后序遍历构造二叉树



    public TreeNode constructMaximumBinaryTree(int[] nums) {
        return constructMaximumBinaryTreeHelp(nums, 0, nums.length);

    }//654最大二叉树
    private TreeNode constructMaximumBinaryTreeHelp(int[] nums, int left, int right){
        if(right - left < 1) return null;//没有元素
        if(right - left == 1) return new TreeNode(nums[left]);//只有一个元素
        int maxIndex = 0;
        int maxVal = 0;
        for(int i = left;i<right;i++){
            if(nums[i] > maxVal){
                maxVal = nums[i];
                maxIndex = i;
            }
        }
        TreeNode node = new TreeNode(maxVal);
        //在max的左右两侧划分子树
        node.left = constructMaximumBinaryTreeHelp(nums, left, maxIndex);
        node.right = constructMaximumBinaryTreeHelp(nums, maxIndex + 1, right);
        return node;

    }

    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        return  mergeTreesHelp1(root1, root2);

    }//617.合并二叉树

    private TreeNode mergeTreesHelp1(TreeNode node1, TreeNode node2){
        if(node1 == null) return node2;
        if(node2 == null) return node1;
        node1.val+= node2.val;
        node1.left = mergeTreesHelp1(node1.left, node2.left);
        node1.right = mergeTreesHelp1(node1.right, node2.right);
        return node1;
    }//递归法

    public TreeNode searchBST(TreeNode root, int val) {
        while (root != null){
            if(root.val == val) return root;
            else if(root.val < val) root = root.right;
            else root = root.left;
        }
        return null;
    }//700二叉搜索树中的搜索

    public boolean isValidBST(TreeNode root) {
        Queue<Integer> que = new LinkedList<>();
        isValidBSTHelp1(root, que);
        int before = que.poll();
        while (!que.isEmpty()){
            int cur = que.poll();
            if(cur <= before) return false;
            before = cur;
        }
        return true;

    }//98验证二叉搜索树
    //将二叉搜索树塞入一个数组
    private void isValidBSTHelp1(TreeNode root, Queue<Integer> que){
        if(root == null) return;
        isValidBSTHelp1(root.left, que);
        que.offer(root.val);
        isValidBSTHelp1(root.right, que);
    }

    public int getMinimumDifference(TreeNode root) {
        Stack<TreeNode> st =new Stack<>();
        TreeNode pre = null;
        int res = Integer.MAX_VALUE;

        if(root == null) return res;
        st.push(root);
        while (!st.isEmpty()){
            TreeNode cur = st.pop();
            if(cur != null){
                if(cur.right!=null) st.push(cur.right);
                st.push(cur);
                st.push(null);
                if(cur.left != null) st.push(cur.left);
            }else {
                TreeNode tem = st.pop();
                if(pre!= null) res = Math.min(res, tem.val - pre.val);
                pre = tem;
            }
        }
        return res;
    }//530二叉搜索树的最小绝对差
    //统一迭代法
    public int[] findMode(TreeNode root) {
        int count = 0;
        int maxCount = 0;
        Map<Integer, Integer> map = new HashMap<>();
        List<Integer> list = new ArrayList<>();
        findModeHelp(root, map);
        for(int key : map.keySet()){
            int val = map.get(key);
           if(val > maxCount){
               maxCount = val;
               list.clear();
               list.add(key);
           }else if(val == maxCount) {
               list.add(key);
           }
        }
        int[] res = new int [list.size()];
        for(int i = 0;i<res.length;i++){
            res[i] = list.get(i);
        }
        return res;


    }//501.二叉搜索树中的众数
    private void findModeHelp(TreeNode root, Map<Integer, Integer> map){
        if(root == null) return;
        map.put(root.val, map.getOrDefault(root.val, 0) + 1);
        findModeHelp(root.left, map);
        findModeHelp(root.right, map);
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null || root == p || root == q){
            return root;
        }
        //后序遍历
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if(left == null && right ==null) return null;
        else if (left != null && right ==null) return left;
        else if(left == null && right != null) return right;
        else return root;
    }//236. 二叉树的最近公共祖先//235. 二叉搜索树的最近公共祖先

    public TreeNode insertIntoBST(TreeNode root, int val) {
         if(root == null) return new TreeNode(val);

         if(root.val < val) root.right = insertIntoBST(root.right, val);
         else if(root.val > val) root.left = insertIntoBST(root.left, val);
         return root;

    }//701二叉搜索树中的插入操作
    public TreeNode deleteNode(TreeNode root, int key) {
        if(root == null) return root;
        if(root.val == key){
            if(root.left == null) return root.right;
            else if(root.right == null) return root.left;
            else {
                TreeNode cur = root.right;
                while (cur.left != null) cur = cur.left;

                cur.left = root.left;
                return root.right;
            }
        }
        if(root.val > key) root.left =deleteNode(root.left, key);
        if(root.val < key) root.right = deleteNode(root.right, key);
        return root;
    }//450删除二叉搜索树中的节点
    public TreeNode trimBST(TreeNode root, int low, int high) {
        if(root == null) return null;
        if(root.val < low) return trimBST(root.right, low, high);
        if(root.val > high) return trimBST(root.left, low, high);

        root.left = trimBST(root.left, low ,high);
        root.right = trimBST(root.right ,low , high);

        return root;
    }//669修建二叉搜索树
    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBSTHelp(nums, 0, nums.length);//左闭右开
    }//108将有序数组转换为二叉搜索树
    private TreeNode sortedArrayToBSTHelp(int[] nums, int left, int right){
        if(left >= right) return null;

        int mid = left + (right - left)/2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = sortedArrayToBSTHelp(nums, left, mid);
        root.right = sortedArrayToBSTHelp(nums, mid+1, right);
        return root;
    }
    public TreeNode convertBST(TreeNode root) {
        int pre = 0;
        Stack<TreeNode> st = new Stack<>();
        if(root == null) return null;

        st.push(root);
        while (!st.isEmpty()){
            TreeNode cur = st.pop();
            if(cur != null){
                if(cur.left != null) st.push(cur.left);//left
                st.push(cur);//mid
                st.push(null);
                if(cur.right != null) st.push(cur.right);//right
            }else {
                TreeNode node = st.pop();
                node.val += pre;
                pre = node.val;
            }
        }
        return root;

    }//538把二叉搜索树转化为累加树

    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        combineHelp1(n,k,1,res,path);
        return res;
    }//77.组合
    private void combineHelp1(int n, int k, int index, List<List<Integer>> res, List<Integer> path){
        if(path.size() == k) {
            //传引用不能这样,后面对path的修改会修改res里面的值res.add(path);
            res.add(new ArrayList<>(path));
            return;
        }
        for(int i = index;i <= n;i++){
            path.add(i);
            combineHelp1(n , k, i+1,res, path);
            path.remove(path.size()-1);
        }
    }

    public List<List<Integer>> combinationSum3(int k, int n) {
        List<Integer> path = new ArrayList<>();
        List<List<Integer>> res = new ArrayList<>();
        combinationgSum3help1(n, k, 1, 0, res, path);
        return res;
    }//216 组合问题三
    private void combinationgSum3help1(int target, int k, int index, int sum, List<List<Integer>> res, List<Integer> path){
        if(sum > target) return;
        if(path.size() == k){
            if (sum == target) res.add(new ArrayList<>(path));
            return;
        }

        for(int i = index; i <= 9 - (k - path.size()) + 1; i++){
            path.add(i);
            sum += i;
            combinationgSum3help1(target, k, i+1, sum, res, path);
            path.remove(path.size()-1);//回溯
            sum -= i;
        }
    }


    public List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();
        List<StringBuilder> resHelp = new ArrayList<>();
        if(digits.length() == 0) return res;
        String[] numsStr = {"","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
        //letterCombinationsHelp1(digits, numsStr, 0,res);
        letterCombinationsHelp2(digits, numsStr, 0,res,new StringBuilder());
        return res;

    }//17.电话号码的字母组合
    private void letterCombinationsHelp1(String digits, String[] numsStr,int num,List<String> res) {
        if(num == digits.length()) return;
        String str = numsStr[digits.charAt(num) - '0'];
        List<StringBuilder> resHelp2 = new ArrayList<>();
        if(res.size() == 0){
            for(char c : str.toCharArray()){
                String temp = "" + c;
                res.add(temp);
            }
        } else for(int i = 0,size = res.size();i < size;i++){
            for(char c : str.toCharArray()){
                String temp = res.get(0) + c;
                res.add(temp);
            }
            res.remove(0);
        }
        letterCombinationsHelp1(digits, numsStr, num+1, res);
    }//遍历法效率低

    private void letterCombinationsHelp2(String digits, String[] numsStr,int num,List<String> res, StringBuilder sb) {
        if (num == digits.length()) {
            res.add(sb.toString());
            return;
        }
        String str = numsStr[digits.charAt(num) - '0'];
        for(char c : str.toCharArray()){
            sb.append(c);
            letterCombinationsHelp2(digits, numsStr, num+1, res, sb);
            sb.deleteCharAt(sb.length() - 1);
        }
    }//回溯法

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        Arrays.sort(candidates);
        combinationSumHelp1(candidates, target,0, 0, path, res);
        //combinationSumHelp2(candidates, target, 0, path, res);//失败的算法

        return res;
    }//39组合总和
    private void combinationSumHelp1(int[] candidates, int target,int index, int sum,List<Integer> path, List<List<Integer>> res) {
        if(sum == target){
            res.add(new ArrayList<>(path));
            return;
        }
        for(int i =index;i< candidates.length;i++){
            if(sum + candidates[i] > target) break;
            path.add(candidates[i]);
            combinationSumHelp1(candidates, target,i, sum + candidates[i], path, res);
            path.remove(path.size()-1);//回溯
        }
    }

   /* private void combinationSumHelp2(int[] candidates, int target, int sum,List<Integer> path, List<List<Integer>> res) {
        sum = 0;
        for (int i: path){
            sum+=i;
        }
        if(sum == target){
            Collections.sort(path);
            if (res.contains(path)) return;
            res.add(new ArrayList<>(path));
            for(int i :path) System.out.print(i + " ");
            System.out.println(sum);
            return;
        }
        for(int i : candidates){
            if(sum + i > target) break;
            path.add(i);
            combinationSumHelp2(candidates, target, sum + i, path, res);
            path.remove(path.size()-1);//回溯
        }
    }

    */


    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        boolean[] used = new boolean[candidates.length];
        Arrays.fill(used, false);
        combinationSum2Help1(candidates, used, target,0,0, res, path);
        return res;
    }//40组合问题2
    private void combinationSum2Help1(int[] candidates, boolean[] used, int target, int index, int sum, List<List<Integer>> res, List<Integer> path) {
        if(sum == target){
            res.add(new ArrayList<>(path));
            return;
        }

        for(int i = index;i < candidates.length;i++){
            if(sum + candidates[i] > target) break;
            if(i > 0 && candidates[i] == candidates[i-1] && !used[i-1]) continue;//同层去重
            used[i] = true;
            path.add(candidates[i]);
            combinationSum2Help1(candidates, used, target, i+1,sum+candidates[i], res, path);
            used[i] = false;
            path.remove(path.size()-1);
        }
    }

    public List<List<String>> partition(String s) {
        List<List<String>> res = new ArrayList<>();
        List<String> path = new ArrayList<>();
        boolean[][] dp = new boolean[s.length() ][s.length()];

        isPalindrome(s,dp);
        partitionHelp1(dp, s, 0, res, path);
        return res;

    }//131分割回文子串
    private void partitionHelp1(boolean[][] dp,String s, int begin,List<List<String>> res,List<String> path){
        if(begin == s.length()){
            res.add(new ArrayList<>(path));
            return;
        }

        for(int i = begin; i < s.length(); i++){
            if(dp[begin][i]){//若果当前子串是回文子串
                path.add(s.substring(begin, i+1));
                partitionHelp1(dp,s,i+1,res,path);
                path.remove(path.size()-1);
            }else {//不是回文子串
                continue;
            }
        }
    }
    private void isPalindrome(String s,boolean[][] dp){
        char []ch = s.toCharArray();
        for(int i = 0;i<ch.length;i++) dp[i][i] = true;
        for(int i = 1;i<ch.length;i++){
            for(int j  = i;j >= 0; j--){
                if(ch[i] == ch[j]){
                    if(i - j <= 1){
                        dp[j][i] = true;
                    }else if(dp[j+1][i-1]){
                        dp[j][i] = true;
                    }
                }
            }
        }
        for(boolean[] test :dp){
            for(boolean test1 : test){
                System.out.print(test1+" ");
            }
            System.out.println();
        }
    }//动态规划给出字符串中的回文子串

    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<>();
        StringBuilder sb = new StringBuilder(s);
        restoreIpAddressesHelp1(sb, 0, 0, res);
        return res;
    }//93复原IP地址
    private void restoreIpAddressesHelp1(StringBuilder s, int index, int dotCount, List<String> res) {
        //终止条件
        if(dotCount == 3){
            if(restoreIpAddressesHelpIsvalid(s, index, s.length() - 1)) {
                res.add(s.toString());
                System.out.println(s.length()-1);
            }
            return;
        }
        for(int i = index; i < s.length(); i++){
            if(restoreIpAddressesHelpIsvalid(s, index, i)){
                s.insert(i+1, '.');
                restoreIpAddressesHelp1(s, i+2, dotCount+1, res);
                s.deleteCharAt(i + 1);//回溯删除刚刚插入的元素
            }else break;

        }
    }
    private boolean restoreIpAddressesHelpIsvalid(StringBuilder s, int start, int end) {
        if(start > end || end - start > 4) return false;
        if(s.charAt(start) == '0' && start != end) return false;

        int num = 0;
        for(int i = start; i <= end; i++){
            num = num * 10 + s.charAt(i) - '0';
        }//这里没有做越界检查,如果num超过int极限也可能过关
        //所以加在前面第一个if了
        return num <= 255;
    }

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        subsetsHelp(nums, 0, res, path);
        return res;
    }//78子集
    private void subsetsHelp(int[] nums, int start, List<List<Integer>> res, List<Integer> path) {
        res.add(new ArrayList<>(path));
        if(start >= nums.length) return;
        for(int i = start; i < nums.length; i++){
            path.add(nums[i]);
            subsetsHelp(nums, i + 1, res, path);
            path.remove(path.size()-1);
        }
    }

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        boolean [] used = new boolean[nums.length];
        //subsetsWithDupHelp1(nums, 0, used, res, path);
        subsetsWithDupHelp2(nums, 0, res, path);
        return res;
    }//90子集,带重复元素的

    private void subsetsWithDupHelp1(int[] nums, int start, boolean[] used, List<List<Integer>> res, List<Integer> path) {
        res.add(new ArrayList<>(path));
        if(start >= nums.length) return;
        for(int i = start; i < nums.length; i++){
            if(i > start && nums[i] == nums[i-1] && !used[i-1]) continue;
            path.add(nums[i]);
            used[i] = true;
            subsetsWithDupHelp1(nums, i + 1, used, res, path);
            path.remove(path.size()-1);//回溯
            used[i] = false;//回溯
        }
    }//使用used数组

    private void subsetsWithDupHelp2(int[] nums, int start, List<List<Integer>> res, List<Integer> path) {
        res.add(new ArrayList<>(path));
        if(start >= nums.length) return;
        for(int i = start; i < nums.length; i++){
            if(i > start && nums[i] == nums[i - 1]) continue;
            path.add(nums[i]);
            subsetsWithDupHelp2(nums, i + 1, res, path);//递归
            path.remove(path.size()-1);//回溯
        }
    }//不使用used数组

    public List<List<Integer>> findSubsequences(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();

        findSubsequencesHelp(nums, 0, res, path);
        return res;
    }//491.非递减子序列
    private void findSubsequencesHelp(int[] nums, int start, List<List<Integer>> res, List<Integer> path) {
        if(path.size() >= 2) {
            res.add(new ArrayList<>(path));
        }
        HashSet<Integer> used = new HashSet<>();
        if(start >= nums.length) return;
        for(int i = start; i < nums.length; i++){
            if(used.contains(nums[i]) ) continue;
            if(path.isEmpty() || nums[i] >= path.get(path.size() - 1)){
                used.add(nums[i]);
                path.add(nums[i]);
                findSubsequencesHelp(nums, i + 1, res, path);
                path.remove(path.size()-1);
            }
        }
    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        permuteHelp(nums, res, path);
        return res;
    }//46.全排列
    private void permuteHelp(int[] nums, List<List<Integer>> res, List<Integer> path) {
        if(path.size() == nums.length) {
            res.add(new ArrayList<>(path));
        }
        for (int num : nums) {
            if (path.contains(num)) continue;
            path.add(num);
            permuteHelp(nums, res, path);
            path.remove(path.size() - 1);
        }
    }

    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        Arrays.sort(nums);
        permuteUniqueHelp1(nums, new boolean[nums.length], res, path);//使用used数组操作,效率更高
        //permuteUniqueHelp2(nums, new boolean[nums.length], res, path);//使用hashset对path头进行过滤,
        return res;

    }//47.全排列2,过滤重复排列
    private void permuteUniqueHelp1(int[] nums, boolean[] used, List<List<Integer>> res, List<Integer> path) {
        if(path.size() == nums.length) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if(i > 0 && nums[i] == nums[i-1] && used[i-1] == false) continue;
            if(used[i] == false) {
                path.add(nums[i]);
                used[i] = true;
                permuteUniqueHelp1(nums, used, res, path);
                path.remove(path.size()-1);
                used[i] = false;
            }
        }

    }
    private void permuteUniqueHelp2(int[] nums, boolean[] used,List<List<Integer>> res, List<Integer> path) {
        if(path.size() == nums.length) {
            res.add(new ArrayList<>(path));
            return;
        }
        HashSet<Integer> set = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            if (set.contains(nums[i])) continue;
            if(used[i] == false){
                path.add(nums[i]);
                set.add(nums[i]);
                used[i] = true;
                permuteUniqueHelp2(nums, used, res, path);
                path.remove(path.size() - 1);
                used[i] = false;
            }

        }
    }


    }

