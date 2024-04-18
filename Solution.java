package bupt;

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


    }
