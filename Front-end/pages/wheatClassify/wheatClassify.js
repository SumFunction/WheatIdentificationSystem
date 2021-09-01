// pages/demo/demo01.js
Page({

  /**
   * 页面的初始数据
   */
  
    data: {
      // 1张图片
      ip:"",
      picturePathLocal:"",
      showPic: true,
      itemhide:false,
      change:".show1",
      url:"../../1.png",
      loadingHidden: true//控制识别动画效果
    },
    /*********1、上传单张图片******************begin */
    //展示图片
    showImg: function (e) {
      var that = this;
      wx.previewImage({
        urls: that.data.picturePathLocal, 
      })
    },
    // 删除图片
    clearImg: function (e) {
      let that = this;
      that.setData({
        showPic: true,
        itemhide:false,
        picturePathLocal:""
      })
    },
  
    //选择图片
    choosePic: function (e) {
      var that = this;
      wx.chooseImage({
        count: 1, // 默认1
        sizeType: ['original', 'compressed'], // 可以指定是原图还是压缩图，默认二者都有
        sourceType: ['album', 'camera'], // 可以指定来源是相册还是相机，默认二者都有
        success: function (res) {
          console.log(res)
            that.setData({
              showPic: false,
              itemhide:true,
              picturePathLocal:res.tempFilePaths[0]
            })
        }
      })
    },
    /**
     * 上传图片
     * @param {*} e 
     */
    upload: function (e) {
      var that = this;
      this.setData({
        loadingHidden: false//控制识别动画效果
      })
      wx.uploadFile({
        url: this.data.ip + '/uploadResnet', //仅为示例，非真实的接口地址
        filePath: that.data.picturePathLocal,
        name: 'file',
        formData: {
          'user': 'test'
        },
        success (res){
          that.setData({
            loadingHidden: true//控制识别动画效果
          })
          var Data = JSON.parse(res.data);
          wx.navigateTo({
            url: '/pages/classifyDetails/classifyDetails?index='+Data.index + '&acc=' + Data.acc
          })
          
        },fail(e){
          that.setData({
            loadingHidden:  true//控制识别动画效果
          })
          wx.showToast({
            title: '未上传文件',
            image:"../../images/cha.png",
            duration: 2000
          })
        }
  
      })
    },
    /*********1、上传单张图片******************end */


  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    var ip = require("../../config");
    this.setData({
      ip:ip.one
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

  }
})