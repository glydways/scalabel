/* global sprintf module*/

/* exported Sat SatImage SatLabel ImageLabel */

if (typeof module !== 'undefined' && typeof module.exports !== 'undefined') {
  module.exports = SatLabel;
}

/*
 Utilities
 */

let COLOR_PALETTE = [
  [31, 119, 180],
  [174, 199, 232],
  [255, 127, 14],
  [255, 187, 120],
  [44, 160, 44],
  [152, 223, 138],
  [214, 39, 40],
  [255, 152, 150],
  [148, 103, 189],
  [197, 176, 213],
  [140, 86, 75],
  [196, 156, 148],
  [227, 119, 194],
  [247, 182, 210],
  [127, 127, 127],
  [199, 199, 199],
  [188, 189, 34],
  [219, 219, 141],
  [23, 190, 207],
  [158, 218, 229],
];

/**
 * Summary: Tune the shade or tint of rgb color
 * @param {[number,number,number]} rgb: input color
 * @param {[number,number,number]} base: base color (white or black)
 * @param {number} ratio: blending ratio
 * @return {[number,number,number]}
 */
function blendColor(rgb, base, ratio) {
  let newRgb = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    newRgb[i] = Math.max(0,
      Math.min(255, rgb[i] + Math.round((base[i] - rgb[i]) * ratio)));
  }
  return newRgb;
}

/**
 * Pick color from the palette. Add additional shades and tints to increase
 * the color number. Results: https://jsfiddle.net/739397/e980vft0/
 * @param {[int]} index: palette index
 * @return {[number,number,number]}
 */
function pickColorPalette(index) {
  let colorIndex = index % COLOR_PALETTE.length;
  let shadeIndex = (Math.floor(index / COLOR_PALETTE.length)) % 3;
  let rgb = COLOR_PALETTE[colorIndex];
  if (shadeIndex === 1) {
    rgb = blendColor(rgb, [255, 255, 255], 0.4);
  } else if (shadeIndex === 2) {
    rgb = blendColor(rgb, [0, 0, 0], 0.2);
  }
  return rgb;
}

/**
 * Base class for each labeling session/task
 * @param {SatItem} ItemType: item instantiation type
 * @param {SatLabel} LabelType: label instantiation type
 */
function Sat(ItemType, LabelType) {
  this.items = []; // a.k.a ImageList, but can be 3D model list
  this.labels = []; // list of label objects
  this.labelIdMap = {};
  this.lastLabelId = -1;
  this.currentItem = null;
  this.ItemType = ItemType;
  this.LabelType = LabelType;
  this.events = [];
  this.startTime = Date.now();
  this.taskId = null;
  this.projectName = null;
  this.ready = false;
  this.getIpInfo();
}

/**
 * Store IP information describing the user using the freegeoip service.
 */
Sat.prototype.getIpInfo = function() {
  let self = this;
  $.getJSON('http://freegeoip.net/json/?callback=?', function(data) {
    self.ipInfo = data;
  });
};

/**
 * Create a new item for this SAT.
 * @param {string} url - Location of the new item.
 * @return {SatItem} - The new item.
 */
Sat.prototype.newItem = function(url) {
  let self = this;
  let item = new self.ItemType(self, self.items.length, url);
  self.items.push(item);
  return item;
};

/**
 * Get a new label ID.
 * @return {int} - The new label ID.
 */
Sat.prototype.newLabelId = function() {
  let newId = this.lastLabelId + 1;
  while (newId in this.labelIdMap) {
    newId += 1;
  }
  this.lastLabelId = newId;
  return newId;
};

/**
 * Create a new label for this SAT.
 * @param {object} optionalAttributes - Optional attributes that may be used by
 *   subclasses of SatLabel.
 * @return {SatLabel} - The new label.
 */
Sat.prototype.newLabel = function(optionalAttributes) {
  let self = this;
  let label = new self.LabelType(self, self.newLabelId(), optionalAttributes);
  self.labelIdMap[label.id] = label;
  self.labels.push(label);
  if (self.currentItem) {
    self.currentItem.labels.push(label);
  }
  return label;
};

/**
 * Add an event to this SAT.
 * @param {string} action - The action triggering the event.
 * @param {int} itemIndex - Index of the item on which the event occurred.
 * @param {int} labelId - ID of the label pertaining to the event.
 * @param {object} position - Object storing some representation of position at
 *   which this event occurred.
 */
Sat.prototype.addEvent = function(action, itemIndex, labelId = -1,
                  position = null) {
  this.events.push({
    timestamp: Date.now(),
    action: action,
    itemIndex: itemIndex,
    labelId: labelId,
    position: position,
  });
};

/**
 * Go to an item in this SAT, setting it to active.
 * @param {int} index - Index of the item to go to.
 */
Sat.prototype.gotoItem = function(index) {
  // mod the index to wrap around the list
  let self = this;
  index = (index + self.items.length) % self.items.length;
  // TODO: event?
  self.currentItem.setActive(false);
  self.currentItem = self.items[index];
  self.currentItem.setActive(true);
  self.currentItem.onload = function() {
    self.currentItem.redraw();
  };
  self.currentItem.redraw();
};

Sat.prototype.loaded = function() {
    this.ready = true;
};

/**
 * Load this SAT from the back end.
 */
Sat.prototype.load = function() {
  let self = this;
  let xhr = new XMLHttpRequest();
  xhr.onreadystatechange = function() {
    if (xhr.readyState === 4) {
      let json = JSON.parse(xhr.response);
      self.fromJson(json);
      self.loaded();
    }
  };
  // get params from url path. These uniquely identify a SAT.
  let searchParams = new URLSearchParams(window.location.search);
  self.taskIndex = parseInt(searchParams.get('task_index'));
  self.projectName = searchParams.get('project_name');
  // send the request to the back end
  let request = JSON.stringify({
    'index': self.taskIndex,
    'projectName': self.projectName,
  });
  xhr.open('POST', './postLoadTask', false);
  xhr.send(request);
};

/**
 * Save this labeling session to file by sending JSON to the back end.
 */
Sat.prototype.save = function() {
  let self = this;
  let json = self.toJson();
  let xhr = new XMLHttpRequest();
  xhr.open('POST', './postSave');
  xhr.send(JSON.stringify(json));
};

/**
 * Get this session's JSON representation
 * @return {{items: Array, labels: Array, events: *, userAgent: string}}
 */
Sat.prototype.toJson = function() {
  let self = this;
  return self.encodeBaseJson();
};

/**
 * Encode the base SAT objects. This should NOT be overloaded. Instead,
 * overload Sat.prototype.toJson()
 * @return {object} - JSON representation of the base functionality in this
 *   SAT. */
Sat.prototype.encodeBaseJson = function() {
  let self = this;
  let items = [];
  for (let i = 0; i < self.items.length; i++) {
    items.push(self.items[i].toJson());
  }
  let labels = [];
  for (let i = 0; i < self.labels.length; i++) {
    if (self.labels[i].valid) {
      labels.push(self.labels[i].toJson());
    }
  }
  return {
    projectName: self.projectName,
    startTime: self.startTime,
    items: items,
    labels: labels,
    categories: self.categories,
    events: self.events,
    userAgent: navigator.userAgent,
    ipInfo: self.ipInfo,
  };
};

/**
 * Initialize this session from a JSON representation
 * @param {string} json - The JSON representation.
 */
Sat.prototype.fromJson = function(json) {
  let self = this;
  self.decodeBaseJson(json);
};

/**
 * Decode the base SAT objects. This should NOT be overloaded. Instead,
 * overload Sat.prototype.fromJson()
 * @param {string} json - The JSON representation.
 */
Sat.prototype.decodeBaseJson = function(json) {
  let self = this;
  for (let i = 0; json.labels && i < json.labels.length; i++) {
    // keep track of highest label ID
    self.lastLabelId = Math.max(self.lastLabelId, json.labels[i].id);
    let newLabel = new self.LabelType(self, json.labels[i].id);
    newLabel.fromJsonVariables(json.labels[i]);
    self.labelIdMap[newLabel.id] = newLabel;
    self.labels.push(newLabel);
  }

  for (let i = 0; i < json.items.length; i++) {
    let newItem = self.newItem(json.items[i].url);
    newItem.fromJson(json.items[i]);
  }

  self.categories = json.category;
  self.assignmentId = json.assignmentId;
  self.projectName = json.projectName;

  self.currentItem = self.items[0];
  self.currentItem.setActive(true);
  self.categories = json.categories;

  for (let i = 0; json.labels && i < json.labels.length; i++) {
    self.labelIdMap[json.labels[i].id].fromJsonPointers(json.labels[i]);
  }
  self.addEvent('start labeling', self.currentItem.index);
};

/**
 * Information used for submission
 * @return {{items: Array, labels: Array, events: *, userAgent: string}}
 */
Sat.prototype.getInfo = function() {
  let self = this;
  let items = [];
  for (let i = 0; i < this.items.length; i++) {
    items.push(this.items[i].toJson());
  }
  let labels = [];
  for (let i = 0; i < this.labels.length; i++) {
    if (this.labels[i].valid) {
      labels.push(this.labels[i].toJson());
    }
  }
  return {
    startTime: self.startTime,
    items: items,
    labels: labels,
    events: self.events,
    userAgent: navigator.userAgent,
    ipAddress: self.ipAddress,
  };
};

/**
 * Base class for each labeling target, can be pointcloud or 2D image
 * @param {Sat} sat: context
 * @param {number} index: index of this item in sat
 * @param {string} url: url to load the item
 */
function SatItem(sat, index = -1, url = '') {
  let self = this;
  self.sat = sat;
  self.index = index;
  self.url = url;
  self.labels = [];
  self.ready = false; // is this needed?
}

SatItem.prototype.setActive = function(active) {
  let self = this;
  if (active) {
    self.sat.addEvent('start labeling', self.index);
  } else {
    self.sat.addEvent('end labeling', self.index);
  }
};

/**
 * Called when this item is loaded.
 */
SatItem.prototype.loaded = function() {
  this.ready = true;
  this.sat.addEvent('loaded', this.index);
};

/**
 * Get the item before this one.
 * @return {SatItem} the item before this one
 */
SatItem.prototype.previousItem = function() {
  if (this.index === 0) {
    return null;
  }
  return this.sat.items[this.index-1];
};

/**
 * Get the SatItem after this one.
 * @return {SatItem} the item after this one
 */
SatItem.prototype.nextItem = function() {
  if (this.index + 1 >= this.sat.items.length) {
    return null;
  }
  return this.sat.items[this.index+1];
};

/**
 * Get this SatItem's JSON representation.
 * @return {object} JSON representation of this item
 */
SatItem.prototype.toJson = function() {
  let self = this;
  let labelIds = [];
  for (let i = 0; i < self.labels.length; i++) {
    if (self.labels[i].valid) {
      labelIds.push(self.labels[i].id);
    }
  }
  return {url: self.url, index: self.index, labelIds: labelIds};
};

/**
 * Restore this SatItem from JSON.
 * @param {object} selfJson - JSON representation of this SatItem.
 * @param {string} selfJson.url - This SatItem's url.
 * @param {number} selfJson.index - This SatItem's index in
 * @param {list} selfJson.labelIds - The list of label ids of this SatItem's
 *   SatLabels.
 */
SatItem.prototype.fromJson = function(selfJson) {
  let self = this;
  self.url = selfJson.url;
  self.index = selfJson.index;
  if (selfJson.labelIds) {
    for (let i = 0; i < selfJson.labelIds.length; i++) {
      self.labels.push(self.sat.labelIdMap[selfJson.labelIds[i]]);
    }
  }
  self.attributes = selfJson.attributes;
};

/**
 * Get all the visible labels in this SatItem.
 * @return {Array} list of all visible labels in this SatItem
 */
SatItem.prototype.getVisibleLabels = function() {
  let labels = [];
  for (let i = 0; i < this.labels.length; i++) {
    if (this.labels[i].valid && this.labels[i].numChildren === 0) {
      labels.push(this.labels[i]);
    }
  }
  return labels;
};

// TODO: remove this function
SatItem.prototype.deleteLabelById = function(labelId, back = true) {
  // TODO: refactor this ugly code
  let self = this;
  for (let i = 0; i < self.labels.length; i++) {
    if (self.labels[i].id === labelId) {
      let currentItem = self.previousItem();
      let currentLabel = self.sat.labelIdMap[self.labels[i].previousLabelId];
      while (back && currentItem) {
        for (let j = 0; j < currentItem.labels.length; j++) {
          if (currentItem.labels[j].id === currentLabel.id) {
            currentItem.labels.splice(j, 1);
            if (currentItem.selectedLabel &&
              currentItem.selectedLabel.id === currentLabel.id) {
              currentItem.selectedLabel = null;
            }
          }
        }
        if (currentLabel) {
          currentLabel = self.sat.labelIdMap[currentLabel.previousLabelId];
        }
        currentItem = currentItem.previousItem();
      }
      currentItem = self.nextItem();
      currentLabel = self.sat.labelIdMap[self.labels[i].nextLabelId];
      while (currentItem) {
        for (let j = 0; j < currentItem.labels.length; j++) {
          if (currentItem.labels[j].id === currentLabel.id) {
            currentItem.labels.splice(j, 1);
            if (currentItem.selectedLabel &&
              currentItem.selectedLabel.id === currentLabel.id) {
              currentItem.selectedLabel = null;
            }
          }
        }
        if (currentLabel) {
          currentLabel = self.sat.labelIdMap[currentLabel.nextLabelId];
        }
        currentItem = currentItem.nextItem();
      }
      self.labels.splice(i, 1);
      if (self.selectedLabel && self.selectedLabel.id === labelId) {
        self.selectedLabel = null;
      }
      return;
    }
  }
};

/**
 * Base class for each targeted labeling Image.
 *
 * To define a new tool:
 *
 * function NewTool() {
 *   SatImage.call(this, sat, index, url);
 * }
 *
 * NewTool.prototype = Object.create(SatImage.prototype);
 *
 * @param {Sat} sat: context
 * @param {number} index: index of this item in sat
 * @param {string} url: url to load the item
 */
function SatImage(sat, index, url) {
  let self = this;
  SatItem.call(self, sat, index, url);
  self.image = new Image();
  self.image.onload = function() {
    self.loaded();
  };
  self.image.src = self.url;

  self.divCanvas = document.getElementById('div_canvas');
  self.imageCanvas = document.getElementById('image_canvas');
  self.hiddenCanvas = document.getElementById('hidden_canvas');
  self.mainCtx = self.imageCanvas.getContext('2d');
  self.hiddenCtx = self.hiddenCanvas.getContext('2d');

  self.imageHeight = self.imageCanvas.height;
  self.imageWidth = self.imageCanvas.width;
  self.hoverLabel = null;

  self.MAX_SCALE = 3.0;
  self.MIN_SCALE = 1.0;
  self.SCALE_RATIO = 1.5;

  self._isMouseDown = false;
}

SatImage.prototype = Object.create(SatItem.prototype);

SatImage.prototype.transformPoints = function(points) {
  let self = this;
  if (points) {
    for (let i = 0; i < points.length; i++) {
      points[i] = points[i] * self.scale;
    }
  }
  return points;
};

/**
 * Set the scale of the image in the display
 * @param {float} scale
 */
SatImage.prototype.setScale = function(scale) {
  let self = this;
  // set scale
  if (scale >= self.MIN_SCALE && scale < self.MAX_SCALE) {
    self.scale = scale;
  } else {
    return;
  }
  // handle buttons
  if (self.scale >= self.MIN_SCALE * self.SCALE_RATIO) {
    $('#decrease_btn').attr('disabled', false);
  } else {
    $('#decrease_btn').attr('disabled', true);
  }
  if (self.scale <= self.MAX_SCALE / self.SCALE_RATIO) {
    $('#increase_btn').attr('disabled', false);
  } else {
    $('#increase_btn').attr('disabled', true);
  }
  // resize canvas
  self.imageCanvas.height = self.imageHeight * self.scale;
  self.imageCanvas.width = self.imageWidth * self.scale;
  self.hiddenCanvas.height = self.imageHeight * self.scale;
  self.hiddenCanvas.width = self.imageWidth * self.scale;
};

SatImage.prototype.loaded = function() {
  // Call SatItem loaded
  SatItem.prototype.loaded.call(this);
};

/**
 * Set whether this SatImage is the active one in the sat instance.
 * @param {boolean} active: if this SatImage is active
 */
SatImage.prototype.setActive = function(active) {
  SatItem.prototype.setActive.call(this);
  let self = this;
  let removeBtn = $('#remove_btn');
  let deleteBtn = $('#delete_btn');
  let endBtn = $('#end_btn');
  if (active) {
    self.imageCanvas = document.getElementById('image_canvas');
    self.hiddenCanvas = document.getElementById('hidden_canvas');
    self.mainCtx = self.imageCanvas.getContext('2d');
    self.hiddenCtx = self.hiddenCanvas.getContext('2d');
    self.state = 'free';
    self.lastLabelID = -1;
    self.padBox = self._getPadding();
    self.catSel = document.getElementById('category_select');
    self.catSel.selectedIndex = 0;
    self.occlCheckbox = document.getElementById('occluded_checkbox');
    self.truncCheckbox = document.getElementById('truncated_checkbox');
    document.getElementById('prev_btn').onclick = function() {
      self.sat.gotoItem(self.index - 1);
    };
    document.getElementById('next_btn').onclick = function() {
      self.sat.gotoItem(self.index + 1);
    };

    // there may be some tension between this and the above block of code
    // TODO: test
    self.imageCanvas.height = self.imageHeight;
    self.imageCanvas.width = self.imageWidth;
    self.hiddenCanvas.height = self.imageHeight;
    self.hiddenCanvas.width = self.imageWidth;
    self.setScale(self.MIN_SCALE);

    // global listeners
    document.onmousedown = function(e) {
      self._mousedown(e);
    };
    document.onmousemove = function(e) {
      self._mousemove(e);
    };
    document.onmouseup = function(e) {
      self._mouseup(e);
    };
    document.onscroll = function(e) {
      self._scroll(e);
    };

    // buttons
    document.getElementById('prev_btn').onclick = function() {
      self.sat.gotoItem(self.index - 1);
    };
    document.getElementById('next_btn').onclick = function() {
      self.sat.gotoItem(self.index + 1);
    };
    if (document.getElementById('increase_btn')) {
      document.getElementById('increase_btn').onclick = function() {
        self._incHandler();
      };
    }
    if (document.getElementById('decrease_btn')) {
      document.getElementById('decrease_btn').onclick = function() {
        self._decHandler();
      };
    }
    if (endBtn.length) {
      // if the end button exists (we have a sequence) then hook it up
      endBtn.click(function() {
        if (self.selectedLabel) {
          self.deleteLabelById(self.selectedLabel.id, false);
          self.redraw();
        }
      });
    }
    if (deleteBtn.length) {
      deleteBtn.click(function() {
        if (self.selectedLabel) {
          self.deleteLabelById(self.selectedLabel.id);
          self.redraw();
        }
      });
    }
    if (removeBtn.length) {
      removeBtn.click(function() {
        if (self.selectedLabel) {
          self.deleteLabelById(self.selectedLabel.id);
          self.redraw();
        }
      });
    }

    // toolbox
    self.catSel = document.getElementById('category_select');
    self.catSel.selectedIndex = 0;
    self.occlCheckbox = document.getElementById('occluded_checkbox');
    self.truncCheckbox = document.getElementById('truncated_checkbox');

    $('#category_select').change(function() {
      self._changeCat();
    });
    $('[name=\'occluded-checkbox\']').on('switchChange.bootstrapSwitch',
      function() {
        self._occlSwitch();
      });
    $('[name=\'truncated-checkbox\']').on('switchChange.bootstrapSwitch',
      function() {
        self._truncSwitch();
      });

    // TODO: Wenqi
    // traffic light color

    self.lastLabelID = 0;
    self.padBox = self._getPadding();
  } else {
    // .click just adds a function to a list of functions that get executed,
    // therefore we need to turn off the old functions
    if (endBtn.length) {
      endBtn.off();
    }
    if (deleteBtn.length) {
      deleteBtn.off();
    }
    if (removeBtn.length) {
      removeBtn.off();
    }
  }
};

/**
 * Redraws this SatImage and all labels.
 */
SatImage.prototype.redraw = function() {
  let self = this;
  self.padBox = self._getPadding();
  self.mainCtx.clearRect(0, 0, self.imageCanvas.width,
    self.imageCanvas.height);
  self.hiddenCtx.clearRect(0, 0, self.hiddenCanvas.width,
    self.hiddenCanvas.height);
  self.mainCtx.drawImage(self.image, 0, 0, self.image.width, self.image.height,
    self.padBox.x, self.padBox.y, self.padBox.w, self.padBox.h);
  for (let i = 0; i < self.labels.length; i++) {
    self.labels[i].redraw(self.mainCtx, self.hiddenCtx, self.selectedLabel,
      self.hoverLabel, i);
  }
};

/**
 * Function to draw the crosshair
 * @param {object} e: mouse event
 */
SatImage.prototype.drawCrossHair = function(e) {
  let divRect = this.divCanvas.getBoundingClientRect();
  let cH = $('#crosshair-h');
  let cV = $('#crosshair-v');
  cH.css('top', e.clientY);
  cH.css('left', divRect.x);
  cH.css('width', divRect.width);
  cV.css('left', e.clientX);
  cV.css('top', divRect.y);
  cV.css('height', divRect.height);
  if (this._isWithinFrame(e)) {
    $('.hair').show();
  } else {
    $('.hair').hide();
  }
};

/**
 * Called when this SatImage is active and the mouse is clicked.
 * @param {object} e: mouse event
 */
SatImage.prototype._mousedown = function(e) {
  let self = this;
  let mousePos = self._getMousePos(e);
  if (self._isWithinFrame(e) && self.state === 'free') {
    [self.selectedLabel, self.currHandle] = self._getSelected(mousePos);
    // change checked traits on label selection
    if (self.selectedLabel) {
      self.selectedLabel.currHandle = self.currHandle;
      for (let i = 0; i < self.catSel.options.length; i++) {
        if (self.catSel.options[i].innerHTML ===
          self.selectedLabel.categoryPath) {
          self.catSel.selectedIndex = i;
          break;
        }
      }
      if ($('[name=\'occluded-checkbox\']').prop('checked') !==
        self.selectedLabel.occl) {
        $('[name=\'occluded-checkbox\']').trigger('click');
      }
      if ($('[name=\'truncated-checkbox\']').prop('checked') !==
        self.selectedLabel.trunc) {
        $('[name=\'truncated-checkbox\']').trigger('click');
      }
      // TODO: Wenqi
      // traffic light color
    }

    if (self.selectedLabel && self.currHandle > 0) {
      // if we have a resize handle
      self.state = 'resize';
      self.selectedLabel.state = 'resize';
      self.resizeID = self.selectedLabel.id;
    } else if (self.currHandle === 0 && self.selectedLabel) {
      // if we have a move handle
      self.selectedLabel.movePos = self.selectedLabel.getCurrentPosition();
      self.selectedLabel.moveClickPos = mousePos;
      self.state = 'move';
      self.selectedLabel.state = 'move';
    } else if (!self.selectedLabel) {
      // otherwise, new label
      let cat = self.catSel.options[self.catSel.selectedIndex].innerHTML;
      let occl = self.occlCheckbox.checked;
      let trunc = self.truncCheckbox.checked;
      self.selectedLabel = self.sat.newLabel({
        categoryPath: cat, occl: occl,
        trunc: trunc, mousePos: mousePos,
      });
      self.selectedLabel.state = 'resize';
      self.state = 'resize';
      self.currHandle = self.selectedLabel.INITIAL_HANDLE;
      self.resizeID = self.selectedLabel.id;
    }
  }
  if (!this._isWithinFrame(e)) {
    return;
  }
  self._isMouseDown = true;
  this.redraw();
};

/**
 * Called when this SatImage is active and the mouse is moved.
 * @param {object} e: mouse event
 */
SatImage.prototype._mousemove = function(e) {
  let mousePos = this._getMousePos(e);
  if (this.sat.LabelType.useCrossHair) {
    this.drawCrossHair(e);
  }
  if (this._isWithinFrame(e)) {
    this.imageCanvas.style.cursor = 'auto';
    // hover effect
    let hoverHandle;
    [this.hoverLabel, hoverHandle] = this._getSelected(mousePos);

    if (this.hoverLabel) {
      this.hoverLabel.setCurrHoverHandle(hoverHandle);
    }
    // label specific handling of mousemove
    if (this.selectedLabel) {
      this.selectedLabel.mousemove(e);
    }

    if (this._isMouseDown && this.selectedLabel) {
      this.imageCanvas.style.cursor = this.selectedLabel.getCursorStyle(
          this.selectedLabel.getCurrHandle());
    } else if (!this._isMouseDown && this.hoverLabel) {
      this.imageCanvas.style.cursor = this.hoverLabel.getCursorStyle(
          this.hoverLabel.getCurrHoverHandle());
    } else {
      this.imageCanvas.style.cursor = 'crosshair';
    }
  }
  this.redraw();
};

/**
 * Called when this SatImage is active and the mouse is released.
 * @param {object} _: mouse event (unused)
 */
SatImage.prototype._mouseup = function(_) { // eslint-disable-line
  this._isMouseDown = false;
  this.state = 'free';

  if (this.selectedLabel) {
    // label specific handling of mouseup
    this.selectedLabel.mouseup();
    if (this.selectedLabel.isSmall()) {
      this.deleteLabelById(this.selectedLabel.id);
    }
  }
  this.redraw();
};

/**
 * Called when this SatImage is active and the mouse is scrolled.
 * @param {object} e: mouse event
 */
SatImage.prototype._scroll = function(e) {
  let self = this;
  if (self.sat.LabelType.useCrossHair) {
    self.drawCrossHair(e);
  }
  self.redraw();
};

/**
 * True if mouse is within the image frame (tighter bound than canvas).
 * @param {object} e: mouse event
 * @return {boolean}: whether the mouse is within the image frame
 */
SatImage.prototype._isWithinFrame = function(e) {
  let rect = this.imageCanvas.getBoundingClientRect();
  let withinImage = (this.padBox
      && rect.x + this.padBox.x < e.clientX
      && e.clientX < rect.x + this.padBox.x + this.padBox.w
      && rect.y + this.padBox.y < e.clientY
      && e.clientY < rect.y + this.padBox.y + this.padBox.h);

  let rectDiv = this.divCanvas.getBoundingClientRect();
  let withinDiv = (rectDiv.x < e.clientX
      && e.clientX < rectDiv.x + rectDiv.width
      && rectDiv.y < e.clientY
      && e.clientY < rectDiv.y + rectDiv.height);
  return withinImage && withinDiv;
};

/**
 * Get the mouse position on the canvas in the image coordinates.
 * @param {object} e: mouse event
 * @return {object}: mouse position (x,y) on the canvas
 */
SatImage.prototype._getMousePos = function(e) {
  let self = this;
  let rect = self.imageCanvas.getBoundingClientRect();
  return {x: (e.clientX - rect.x) / self.scale,
    y: (e.clientY - rect.y) / self.scale};
};

/**
 * Get the padding for the image given its size and canvas size.
 * @return {object}: padding box (x,y,w,h)
 */
SatImage.prototype._getPadding = function() {
  // which dim is bigger compared to canvas
  let xRatio = this.image.width / this.imageCanvas.width;
  let yRatio = this.image.height / this.imageCanvas.height;
  // use ratios to determine how to pad
  let box = {x: 0, y: 0, w: 0, h: 0};
  if (xRatio >= yRatio) {
    box.x = 0;
    box.y = 0.5 * (this.imageCanvas.height - this.imageCanvas.width *
      this.image.height / this.image.width);
    box.w = this.imageCanvas.width;
    box.h = this.imageCanvas.height - 2 * box.y;
  } else {
    box.x = 0.5 * (this.imageCanvas.width - this.imageCanvas.height *
      this.image.width / this.image.height);
    box.y = 0;
    box.w = this.imageCanvas.width - 2 * box.x;
    box.h = this.imageCanvas.height;
  }
  return box;
};

/**
 * Get the label with a given id.
 * @param {number} labelID: id of the sought label
 * @return {ImageLabel}: the sought label
 */
SatImage.prototype._getLabelByID = function(labelID) {
  for (let i = 0; i < this.labels.length; i++) {
    if (this.labels[i].id === labelID) {
      return this.labels[i];
    }
  }
};

/**
 * Get the box and handle under the mouse.
 * @param {object} mousePos: canvas mouse position (x,y)
 * @return {[ImageLabel, number]}: the box and handle (0-9) under the mouse
 */
SatImage.prototype._getSelected = function(mousePos) {
  let pixelData = this.hiddenCtx.getImageData(mousePos.x,
    mousePos.y, 1, 1).data;
  let selectedLabelIndex = null;
  let currHandle = null;
  if (pixelData[3] !== 0) {
    selectedLabelIndex = pixelData[0] * 256 + pixelData[1];
    currHandle = pixelData[2] - 1;
  }
  let selectedLabel = this.labels[selectedLabelIndex];
  return [selectedLabel, currHandle];
};


/**
 * Called when the selected category is changed.
 */
SatImage.prototype._changeCat = function() {
  let self = this;
  if (self.selectedLabel) {
    let option = self.catSel.options[self.catSel.selectedIndex].innerHTML;
    self.selectedLabel.categoryPath = option;
    self.redraw();
  }
};

/**
 * Called when the occluded checkbox is toggled.
 */
SatImage.prototype._occlSwitch = function() {
    if (this.selectedLabel) {
        this.occl = $('[name=\'occluded-checkbox\']').prop('checked');
    }
};

/**
 * Called when the truncated checkbox is toggled.
 */
SatImage.prototype._truncSwitch = function() {
    if (this.selectedLabel) {
        this.trunc = $('[name=\'truncated-checkbox\']').prop(
            'checked');
    }
};

/**
 * Increase button handler
 */
SatImage.prototype._incHandler = function() {
  let self = this;
  self.setScale(self.scale * self.SCALE_RATIO);
  self.redraw();
};

/**
 * Decrease button handler
 */
SatImage.prototype._decHandler = function() {
  let self = this;
  self.setScale(self.scale / self.SCALE_RATIO);
  self.redraw();
};

/**
 * Called when the traffic light color choice is changed.
 */
SatImage.prototype._lightSwitch = function() {
    // TODO: Wenqi
};

/**
 * Base class for all the labeled objects. New label should be instantiated by
 * Sat.newLabel()
 *
 * To define a new tool:
 *
 * function NewObject(id) {
 *   SatLabel.call(this, id);
 * }
 *
 * NewObject.prototype = Object.create(SatLabel.prototype);
 *
 * @param {Sat} sat: The labeling session
 * @param {number | null} id: label object identifier
 * @param {object} ignored: ignored parameter for optional attributes.
 */
function SatLabel(sat, id = -1, ignored = null) {
  this.id = id;
  this.categoryPath = null;
  this.attributes = {};
  this.sat = sat;
  this.parent = null;
  this.children = [];
  this.numChildren = 0;
  this.valid = true;
  this.currHandle = 0;
  this.currHoverHandle = 0;
}

SatLabel.prototype.delete = function() {
  this.valid = false;
  if (this.parent !== null) {
    this.parent.numChildren -= 1;
    if (this.parent.numChildren === 0) this.parent.delete();
  }
  for (let i = 0; i < this.children.length; i++) {
    this.children[i].parent = null;
    this.children[i].delete();
  }
};

SatLabel.prototype.setCurrHandle = function(handle) {
  this.currHandle = Math.max(0, handle);
};

SatLabel.prototype.setCurrHoverHandle = function(handle) {
  this.currHoverHandle = Math.max(0, handle);
};

SatLabel.prototype.getCurrHandle = function() {
  return this.currHandle;
};

SatLabel.prototype.getCurrHoverHandle = function() {
  return this.currHoverHandle;
};

SatLabel.prototype.getRoot = function() {
  if (this.parent === null) return this;
  else return this.parent.getRoot();
};

/**
 * Get the current position of this label.
 */
SatLabel.prototype.getCurrentPosition = function() {

};


SatLabel.prototype.addChild = function(child) {
  this.numChildren += 1;
  this.children.push(child);
};

/**
 * Pick a color based on the label id
 * @return {(number|number|number)[]}
 */
SatLabel.prototype.color = function() {
  return pickColorPalette(this.getRoot().id);
};

/**
 * Convert the color to css style
 * @param {number} alpha: color transparency
 * @return {[number,number,number]}
 */
SatLabel.prototype.styleColor = function(alpha = 255) {
  let c = this.color();
  return sprintf('rgba(%d, %d, %d, %f)', c[0], c[1], c[2], alpha);
};

SatLabel.prototype.encodeBaseJson = function() {
  let self = this;
  let json = {id: self.id, categoryPath: self.categoryPath};
  if (self.parent) {
    json.parent = self.parent.id;
  } else {
    json.parent = -1;
  }
  if (self.children && self.children.length > 0) {
    let childrenIds = [];
    for (let i = 0; i < self.children.length; i++) {
      if (self.children[i].valid) {
        childrenIds.push(self.children[i].id);
      }
    }
    json.children = childrenIds;
  }
  json.previousLabelId = -1;
  json.nextLabelId = -1;
  if (self.previousLabelId) {
    json.previousLabelId = self.previousLabelId;
  }
  if (self.nextLabelId) {
    json.nextLabelId = self.nextLabelId;
  }
  // TODO: remove
  json.keyframe = self.keyframe;
  return json;
};

/**
 * Return json object encoding the label information
 * @return {{id: *}}
 */
SatLabel.prototype.toJson = function() {
  let self = this;
  return self.encodeBaseJson();
};

SatLabel.prototype.decodeBaseJsonVariables = function(json) {
  let self = this;
  self.id = json.id;
  self.categoryPath = json.categoryPath;
  // TODO: remove
  self.keyframe = json.keyframe;
  if (json.previousLabelId > -1) {
    self.previousLabelId = json.previousLabelId;
  }
  if (json.nextLabelId > -1) {
    self.nextLabelId = json.nextLabelId;
  }
};

SatLabel.prototype.decodeBaseJsonPointers = function(json) {
  let self = this;
  let labelIdMap = self.sat.labelIdMap;
  labelIdMap[self.id] = self;
  self.sat.lastLabelId = Math.max(self.sat.lastLabelId, self.id);
  if (json.parent > -1) {
    self.parent = labelIdMap[json.parent];
  }

  if (json.children) {
    let childrenIds = json.children;
    for (let i = 0; i < childrenIds.length; i++) {
      self.addChild(labelIdMap[childrenIds[i]]);
    }
  }
};

/**
 * Load label information from json object
 * @param {object} json: JSON representation of this SatLabel.
 */
SatLabel.prototype.fromJsonVariables = function(json) {
  let self = this;
  self.decodeBaseJsonVariables(json);
};

SatLabel.prototype.fromJsonPointers = function(json) {
  let self = this;
  self.decodeBaseJsonPointers(json);
};

SatLabel.prototype.startChange = function() {

};

SatLabel.prototype.updateChange = function() {

};

SatLabel.prototype.finishChange = function() {

};

SatLabel.prototype.redraw = function() {

};

/**
 * Base class for all the labeled objects. New label should be instantiated by
 * Sat.newLabel()
 *
 * To define a new tool:
 *
 * function NewObject(sat, id) {
 *   ImageLabel.call(this, sat, id);
 * }
 *
 * NewObject.prototype = Object.create(ImageLabel.prototype);
 *
 * @param {Sat} sat: The labeling session
 * @param {number | null} id: label object identifier
 * @param {object} optionalAttributes: Optional attributes for the SatLabel.
 */
function ImageLabel(sat, id, optionalAttributes = null) {
  SatLabel.call(this, sat, id, optionalAttributes);
  this.image = sat.currentItem;
}

ImageLabel.prototype = Object.create(SatLabel.prototype);

ImageLabel.useCrossHair = false;

ImageLabel.prototype.getCurrentPosition = function() {

};

ImageLabel.prototype.fromJsonPointers = function(json) {
  let self = this;
  self.decodeBaseJsonPointers(json);
  self.image = self.sat.currentItem;
};

/**
 * Get the weighted average between this label and a provided label.
 * @param {ImageLabel} ignoredLabel - The other label.
 * @param {number} ignoredWeight - The weight, b/w 0 and 1, higher
 * corresponds to
 *   closer to the other label.
 * @return {object} - The label's position.
 */
ImageLabel.prototype.getWeightedAvg = function(ignoredLabel, ignoredWeight) {
  return null;
};

/**
 * Set this label to be the weighted average of the two provided labels.
 * @param {ImageLabel} ignoredStartLabel - The first label.
 * @param {ImageLabel} ignoredEndLabel - The second label.
 * @param {number} ignoredWeight - The weight, b/w 0 and 1, higher
 *   corresponds to closer to endLabel.
 */
ImageLabel.prototype.weightedAvg = function(ignoredStartLabel, ignoredEndLabel,
                                            ignoredWeight) {

};

/**
 * Calculate the intersection between this and another ImageLabel
 * @param {ImageLabel} ignoredLabel - The other image label.
 * @return {number} - The intersection between the two labels.
 */
ImageLabel.prototype.intersection = function(ignoredLabel) {
  return 0;
};

/**
 * Calculate the union between this and another ImageLabel
 * @param {ImageLabel} ignoredLabel - The other image label.
 * @return {number} - The union between the two labels.
 */
ImageLabel.prototype.union = function(ignoredLabel) {
  return 0;
};

/**
 * Draw a specified resize handle of this bounding box.
 * @param {object} ctx - Canvas context.
 * @param {number} handleNo - The handle number, i.e. which handle to draw.
 */
ImageLabel.prototype.drawHandle = function(ctx, handleNo) {
  let self = this;
  ctx.save(); // save the canvas context settings
  let posHandle = self.getHandle(handleNo);

  [posHandle.x, posHandle.y] = self.image.transformPoints(
      [posHandle.x, posHandle.y]);

  if (self.isSmall()) {
    ctx.fillStyle = 'rgb(169, 169, 169)';
  } else {
    ctx.fillStyle = self.styleColor();
  }
  ctx.lineWidth = self.LINE_WIDTH;
  if (posHandle) {
    ctx.beginPath();
    ctx.arc(posHandle.x, posHandle.y, self.HANDLE_RADIUS, 0, 2 * Math.PI);
    ctx.fill();
    if (!self.isSmall()) {
      ctx.strokeStyle = 'white';
      ctx.lineWidth = self.OUTLINE_WIDTH;
      ctx.stroke();
    }
  }
  ctx.restore(); // restore the canvas to saved settings
};