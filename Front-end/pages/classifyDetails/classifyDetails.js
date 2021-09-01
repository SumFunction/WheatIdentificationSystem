// pages/classifyDetails/classifyDetails.js
Page({

  /**
   * 页面的初始数据
   */
  data: {
      acc:"",
      chiName:"",
      engName:"",
      introduce:"",
      feature:"",
      measure:"",
      regular:"",
      time:"",
      curtime:"",
      url1:"",
      url2:""
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    var ip = require("../../config");
    ip = ip.one;
    var jsonData = require("/data.js");
    jsonData = jsonData.dataList;
    var acc = options.acc;
    acc = (acc*100).toFixed(2)
    var index = options.index;
    jsonData= jsonData[index];
    this.setData({
      acc:acc,
      chiName:jsonData["chiName"],
      engName:jsonData["engName"],
      introduce:jsonData["introduce"],
      measure:jsonData["measure"],
      regular:jsonData["regular"],
      feature:jsonData["feature"],
      time:jsonData['time'],
      curtime:jsonData['curtime'],
      url1: jsonData.url1,
      url2: jsonData.url2
    })
  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide: function () {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {

  },
  click_me: function(event) {
    console.log(111)
  }
})