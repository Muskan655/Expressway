import mongoose from "mongoose";
import bcrypt from "bcrypt";

const { Schema } = mongoose;

// ----------------- User Schema -----------------
export const userSchema = new Schema({
  username: { type: String, required: true, unique: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  displayName: { type: String },
  avatarUrl: { type: String },
});


userSchema.pre("save", async function (next) {
  if (!this.isModified("password")) return next();
  const salt = await bcrypt.genSalt(10);
  this.password = await bcrypt.hash(this.password, salt);
  next();
});


userSchema.methods.comparePassword = async function (candidatePassword) {
  return bcrypt.compare(candidatePassword, this.password);
};

// ----------------- Room Schema -----------------
export const roomSchema = new Schema({
  name: { type: String, required: true },
  members: [{ type: String }], 
  pending: [{ type: String }], 
});

// ----------------- Message Schema -----------------
const messageSchema = new Schema({
  roomId: { type: Schema.Types.ObjectId, ref: "Room", required: true },
  sender: { type: String, required: true },
  type: { type: String, required: true },
  content: { type: Schema.Types.Mixed, required: true }, 
  votes: { type: Number, default: 0 },
  voters: { type: [{ username: String, vote: Number }], default: [] }
}, { timestamps: true });

export const eventSchema = new Schema({
  userId: { type: Schema.Types.ObjectId, ref: "User", required: true }, 
  title: { type: String, required: true },
  date: { type: Date, required: true },
  occasion: { type: String, required: true },
  createdAt: { type: Date, default: Date.now }
});


export const Event = mongoose.model("Event", eventSchema);

export const User = mongoose.model("User", userSchema);
export const Room = mongoose.model("Room", roomSchema);
export const Message = mongoose.model("Message", messageSchema);
